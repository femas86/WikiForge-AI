import hashlib
import json
import logging
import os
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pkms import metrics
from pkms.embed import embed
from pkms.guards import guard_write, validate_project
from pkms.ingestor import chunk
from pkms.llm import complete
from pkms.qdrant_store import point_id, search, upsert
from pkms.memory import get_provider
from pkms.user_prefs import load_user_style

logger = logging.getLogger(__name__)

# ── Mem0 helpers (non-fatal wrappers) ─────────────────────────────────────────
# Backed by the managed Mem0 platform (app.mem0.ai). The client reads its
# credential from the MEM0_API_KEY environment variable (loaded from .env by
# web.py / coordinator.py). If the key is absent or the service is unreachable,
# every call degrades silently: recall returns [], store is a no-op.

def _mem0_client():
    """Instantiate the managed Mem0 client. Raises if MEM0_API_KEY is unset."""
    from mem0 import MemoryClient
    return MemoryClient(api_key=os.getenv("MEM0_API_KEY"))


def _mem0_recall(question: str, user_id: str, config: dict[str, Any]) -> list[str]:
    top_k = config["query"]["mem0_recall"]
    try:
        m = _mem0_client()
        # Platform search: entity params go in `filters`, not top-level kwargs;
        # response is v1.1 format {"results": [...]}.
        resp = m.search(question, filters={"user_id": user_id}, top_k=top_k)
        results = resp.get("results", []) if isinstance(resp, dict) else (resp or [])
        return [r.get("memory", "") for r in results if r.get("memory")]
    except Exception as exc:
        logger.warning("Mem0 recall failed (non-fatal): %s", exc)
        return []


def _mem0_store(
    question: str,
    answer_md: str,
    sources: list[str],
    user_id: str,
    session_id: str,
    config: dict[str, Any],
    project: str = "default",
) -> None:
    try:
        m = _mem0_client()
        m.add(
            messages=[
                {"role": "user", "content": question},
                {"role": "assistant", "content": answer_md},
            ],
            user_id=user_id,
            metadata={
                "type": "qa_interaction",
                "project": project,
                "sources": sources,
                "session_id": session_id,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
        )
    except Exception as exc:
        logger.warning("Mem0 store failed (non-fatal): %s", exc)


# ── synthesis prompt ──────────────────────────────────────────────────────────

_SYNTHESIS_PROMPT = """\
You are a research assistant answering questions from a knowledge base.

Question: {question}

Wiki chunks (compiled, curated):
{wiki_chunks}

Raw source chunks (uncompiled):
{raw_chunks}

Wiki index (one-line map of all articles):
{index_md}

Prior context from this user's past questions:
{prior_context}

User's preferred answer style (apply to formatting/tone/language; never let it
override factual grounding or citation rules):
{user_instructions}

Instructions:
- Answer ONLY from the retrieved chunks above. Do not hallucinate.
- Prefer wiki chunks; use raw chunks to fill gaps or if wiki lacks coverage.
- Cite every claim: use [[article-slug]] for wiki sources, or "raw:<path>#<chunk_index>" for raw.
- At the end, include a JSON block exactly like this:
```json
{{"sources": ["..."], "coverage": "full|partial|raw_only|none"}}
```
- coverage values:
  - full: wiki chunks fully answer the question
  - partial: wiki covers most of it but raw chunks needed for some parts
  - raw_only: wiki has nothing relevant; answer drawn entirely from raw chunks
  - none: neither wiki nor raw chunks contain relevant information
- If coverage is none, say "No relevant documents found in the knowledge base."
"""


def _format_hits(hits: list[dict[str, Any]]) -> str:
    if not hits:
        return "(none)"
    lines = []
    for h in hits:
        p = h.get("payload", {})
        path = p.get("path", "")
        idx = p.get("chunk_index", 0)
        heading = p.get("section_heading", "")
        text = p.get("text", "")
        header = f"[{path}#{idx}]" + (f" {heading}" if heading else "")
        lines.append(f"{header}\n{text}")
    return "\n\n".join(lines)


def _parse_llm_response(raw: str) -> tuple[str, list[str], str]:
    """Extract (answer_md, sources, coverage) from LLM output."""
    # Extract trailing JSON block
    match = re.search(r"```json\s*(\{.*?\})\s*```", raw, re.DOTALL)
    if match:
        try:
            meta = json.loads(match.group(1))
            sources = meta.get("sources", [])
            coverage = meta.get("coverage", "partial")
            answer_md = raw[: match.start()].strip()
            return answer_md, sources, coverage
        except json.JSONDecodeError:
            pass
    # Fallback: return full text with unknown coverage
    return raw.strip(), [], "partial"


# ── output write ─────────────────────────────────────────────────────────────

def _write_output(
    answer_md: str,
    question: str,
    sources: list[str],
    coverage: str,
    session_id: str,
    vault_dir: Path,
    config: dict[str, Any],
    project: str,
) -> str:
    """Write answer to vault/{project}/outputs/ and upsert to Qdrant outputs collection."""
    outputs_dir = vault_dir / project / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)

    safe_slug = re.sub(r"[^\w\-]", "_", session_id)[:60]
    filename = f"{safe_slug}.md"
    out_abs = outputs_dir / filename
    vault_rel = f"vault/{project}/outputs/{filename}"

    full_content = f"# Query\n\n{question}\n\n# Answer\n\n{answer_md}\n\n**Coverage:** {coverage}\n\n**Sources:** {', '.join(sources)}\n"

    guard_write("querier", str(out_abs), str(vault_dir))
    out_abs.write_text(full_content, encoding="utf-8")

    # Embed and upsert to outputs collection
    collection = config["qdrant"]["collections"]["outputs"]
    max_tokens = config["chunking"]["max_tokens"]
    timestamp = datetime.now(timezone.utc).isoformat()
    content_hash = "sha256:" + hashlib.sha256(full_content.encode("utf-8")).hexdigest()
    chunks = chunk(full_content, max_tokens)
    for i, ch in enumerate(chunks):
        pid = point_id(vault_rel, i)
        try:
            vector = embed(ch["text"], config)
            upsert(collection, pid, vector, {
                "path": vault_rel,
                "project": project,
                "collection": collection,
                "title": question[:80],
                "summary_1line": question[:80],
                "tags": [],
                "section_heading": ch["section_heading"],
                "chunk_index": i,
                "chunk_total": len(chunks),
                "hash": content_hash,
                "agent": "querier",
                "timestamp": timestamp,
            }, config)
        except Exception as exc:
            logger.warning("Output upsert failed for chunk %d: %s", i, exc)

    return vault_rel


# ── main query function ───────────────────────────────────────────────────────

def query(
    question: str,
    user_id: str,
    vault_root: str,
    config: dict[str, Any],
    session_id: str | None = None,
    prior_context: list[str] | None = None,
    project: str = "default",
) -> dict[str, Any]:
    """Answer a question using dual search over one project's wiki + raw chunks.

    prior_context: list of prior memory strings (passed by Coordinator from Mem0 recall).
    Returns result dict with answer_md, sources, coverage, output_path.
    """
    validate_project(project)
    if session_id is None:
        # uuid suffix: second-granularity timestamps collide when teammates share a user_id
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
        session_id = f"query_{ts}_{user_id}_{uuid.uuid4().hex[:6]}"

    vault_dir = Path(vault_root) / "vault"
    wiki_collection = config["qdrant"]["collections"]["wiki"]
    raw_collection = config["qdrant"]["collections"]["raw"]
    top_k_wiki = config["query"]["top_k_wiki"]
    top_k_raw = config["query"]["top_k_raw"]

    # Metrics: the querier owns the collection scope; llm/embed record their
    # native usage (tokens, durations) into it, phases are timed here. The
    # result carries {"summary", "events"} for observability (per-query token
    # cost and latency, broken down by phase).
    with metrics.collect() as events:
        # Embed question + dual search — always both collections, project-scoped
        with metrics.timer("retrieval", phase="dual_search"):
            query_vector = embed(question, config)
            wiki_hits = search(wiki_collection, query_vector, top_k_wiki, config, project=project)
            raw_hits = search(raw_collection, query_vector, top_k_raw, config, project=project)

        # Load _index.md (non-fatal if missing)
        index_path = vault_dir / project / "wiki" / "_index.md"
        index_md = index_path.read_text(encoding="utf-8") if index_path.exists() else "(wiki index not available)"

        # Memory recall via the configured provider (none|mem0|…), if the
        # Coordinator didn't already pass prior_context.
        provider = get_provider(config)
        if prior_context is None:
            with metrics.timer("memory_recall", provider=provider.name):
                prior_context = provider.recall(question, user_id, config)

        prior_text = "\n".join(prior_context) if prior_context else "(none)"
        user_instructions = load_user_style(user_id) or "(none)"

        # Synthesise answer
        prompt = _SYNTHESIS_PROMPT.format(
            question=question,
            wiki_chunks=_format_hits(wiki_hits),
            raw_chunks=_format_hits(raw_hits),
            index_md=index_md,
            prior_context=prior_text,
            user_instructions=user_instructions,
        )
        with metrics.timer("synthesis"):
            raw_response = complete("querier", prompt, config)
        answer_md, sources, coverage = _parse_llm_response(raw_response)

        # Store interaction via the configured provider. Non-fatality is the
        # provider's contract (see MemoryProvider.store) — every provider wraps its
        # own backend calls; no second net here that would blur misconfiguration
        # (caught loudly at provider construction) with transient failures.
        with metrics.timer("memory_store", provider=provider.name):
            provider.store(question, answer_md, sources, user_id, session_id, config, project=project)

        # Write output file
        output_path = _write_output(
            answer_md, question, sources, coverage, session_id, vault_dir, config, project
        )

    summary = metrics.summarize(events)
    logger.info(
        "Query done: coverage=%s, sources=%d, llm_calls=%d, tokens_in=%d, tokens_out=%d, output=%s",
        coverage, len(sources), summary["llm_calls"],
        summary["tokens_in"], summary["tokens_out"], output_path,
    )

    return {
        "answer_md": answer_md,
        "sources": sources,
        "coverage": coverage,
        "output_path": output_path,
        "session_id": session_id,
        "metrics": {"summary": summary, "events": events},
    }
