import hashlib
import json
import logging
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import httpx

from pkms.db import get_file
from pkms.embed import embed_many
from pkms.guards import guard_write, validate_project
from pkms.ingest_marker import clear_ingesting, mark_ingesting
from pkms.llm import complete, retry_transient
from pkms.indexing import embed_and_upsert_chunks
from pkms.qdrant_store import delete_by_ids

logger = logging.getLogger(__name__)

# ── parsing ───────────────────────────────────────────────────────────────────

def _hash_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(65536), b""):
            h.update(block)
    return f"sha256:{h.hexdigest()}"


def _hash_text(text: str) -> str:
    return "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()


def _parse_text(path: str) -> str:
    return Path(path).read_text(encoding="utf-8", errors="replace")


def _space_ratio(text: str) -> float:
    return text.count(" ") / len(text) if text else 1.0


def _looks_glued(text: str) -> bool:
    """True when extraction ran words together (e.g. 'Neuro-SymbolicAIin2024').

    Normal prose is ~15-18% spaces; a glued pdfminer extraction is far lower.
    Guarded by a length floor so tiny/structured docs don't false-positive."""
    return len(text) > 200 and _space_ratio(text[:5000]) < 0.08


def _parse_pdf(path: str) -> str:
    """pdfminer first; if it glues words, try pypdf and keep the better-spaced text.

    Glued text (B6) wrecks tags and embeddings downstream, so it's worth a second
    extractor rather than indexing a run-on blob."""
    from pdfminer.high_level import extract_text
    text = extract_text(path) or ""
    if _looks_glued(text):
        logger.warning("PDF %s: pdfminer text looks glued (space ratio %.3f) — trying pypdf",
                       path, _space_ratio(text[:5000]))
        alt = _parse_pdf_pypdf(path)
        if alt and _space_ratio(alt) > _space_ratio(text):
            logger.info("PDF %s: using pypdf extraction (better spacing)", path)
            return alt
    return text


def _parse_pdf_pypdf(path: str) -> str:
    try:
        from pypdf import PdfReader
        return "\n".join((page.extract_text() or "") for page in PdfReader(path).pages)
    except Exception as exc:
        logger.warning("pypdf fallback failed for %s: %s", path, exc)
        return ""


def _parse_html(path: str) -> str:
    """Main-content extraction (drops nav/boilerplate/scripts), with fallbacks.

    The old regex tag-strip ingested everything — nav, footers, and inline JS
    (the boilerplate that made scraped pages like infonce.html useless). Prefer
    trafilatura's article extraction; fall back to bs4 with script/style removed;
    last resort, the naive regex strip."""
    raw = Path(path).read_text(encoding="utf-8", errors="replace")

    try:
        import trafilatura
        extracted = trafilatura.extract(raw, include_comments=False, include_tables=True)
        if extracted and extracted.strip():
            return extracted.strip()
        logger.info("trafilatura found no main content in %s — falling back", path)
    except Exception as exc:
        logger.warning("trafilatura extraction failed for %s: %s", path, exc)

    try:
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(raw, "html.parser")
        for tag in soup(["script", "style", "noscript", "head", "nav", "header", "footer"]):
            tag.decompose()
        return re.sub(r"\s+", " ", soup.get_text(" ")).strip()
    except Exception as exc:
        logger.warning("bs4 HTML parse failed for %s: %s", path, exc)

    text = re.sub(r"<[^>]+>", " ", raw)          # last resort
    return re.sub(r"\s+", " ", text).strip()


def parse(path: str) -> str:
    ext = Path(path).suffix.lower()
    if ext == ".pdf":
        return _parse_pdf(path)
    if ext == ".html":
        return _parse_html(path)
    return _parse_text(path)  # .md, .txt, fallback


# ── metadata extraction ───────────────────────────────────────────────────────

_METADATA_PROMPT = """\
Extract metadata from the document excerpt below. Reply with a JSON object only — no explanation.

Required fields:
- "title": string (document title or best guess)
- "summary_1line": string (one sentence summary, max 20 words)
- "tags": array of strings (3–8 lowercase topic tags)
- "doc_type": one of "paper", "note", "html", "pdf", "markdown"

Document excerpt:
{sample}
"""


def extract_metadata(text: str, path: str, config: dict[str, Any]) -> dict[str, Any]:
    sample = text[:3000]
    prompt = _METADATA_PROMPT.format(sample=sample)
    ing = config.get("ingest", {})
    temp = ing.get("metadata_temperature", 0.1)
    model = ing.get("metadata_model")            # small/fast Ollama model; None → configured default
    num_predict = ing.get("metadata_num_predict", 256)
    try:
        # retries=1: fail fast to the Claude fallback instead of burning ~6 min of Ollama timeouts.
        # response_format="json": constrain Ollama to valid JSON (small models like
        # llama3.2:1b otherwise emit unparseable output → empty tags/default title).
        raw = complete("ingestor", prompt, config, temperature=temp,
                       model=model, num_predict=num_predict, retries=1,
                       response_format="json")
    except Exception as exc:
        # Both the primary backend and its fallback failed. Metadata is
        # nice-to-have, not essential — never let it kill the ingest; fall back
        # to filename-derived defaults below.
        logger.warning("Metadata LLM failed for %s (%s); using defaults", path, exc)
        raw = ""
    # Strip markdown code fences if present
    raw = re.sub(r"^```[^\n]*\n?", "", raw.strip())
    raw = re.sub(r"```$", "", raw.strip())
    try:
        meta = json.loads(raw)
    except json.JSONDecodeError:
        logger.warning("Metadata JSON parse failed for %s; using defaults", path)
        meta = {}
    # Ensure all required keys are present with safe defaults
    ext = Path(path).suffix.lstrip(".")
    meta.setdefault("title", Path(path).stem)
    meta.setdefault("summary_1line", "")
    meta.setdefault("tags", [])
    meta.setdefault("doc_type", ext if ext in ("pdf", "html", "markdown") else "note")
    return meta


# ── chunking ──────────────────────────────────────────────────────────────────

def _estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def _detect_heading(text: str) -> str:
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        # Markdown headings
        if stripped.startswith("#"):
            return stripped.lstrip("#").strip()
        # Heuristic for PDF-extracted text: short isolated lines that look like
        # section titles (ALL CAPS or Title Case, no trailing sentence punctuation,
        # 1–10 words). Catches "Abstract", "INTRODUCTION", "3. Model Architecture".
        if (
            1 <= len(stripped.split()) <= 10
            and len(stripped) <= 80
            and stripped[-1] not in ".,:;!?)"
            and (stripped.isupper() or stripped.istitle())
        ):
            return stripped
    return ""


def _char_split(text: str, max_tokens: int) -> list[str]:
    """Blind hard-split on a character boundary (~max_tokens*4 chars).

    The last-resort safety net so no chunk exceeds the embedder's context: some
    PDFs extract as one blob with no sentence/blank-line structure to split on.
    """
    max_chars = max(1, max_tokens * 4)  # _estimate_tokens ≈ len // 4
    return [text[i:i + max_chars] for i in range(0, len(text), max_chars)]


def chunk(text: str, max_tokens: int, config: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    """Split text into chunks. Strategy comes from config["chunking"]["strategy"].

    - "fixed" (default when no config): paragraph-merge to max_tokens; an oversized
      single paragraph is hard-split on a character boundary.
    - "semantic": small paragraphs still merge structurally, but any WIDE paragraph
      (>= semantic_min_tokens — a candidate for holding several key concepts) is
      split at meaning-aware sentence boundaries via sentence embeddings, so a
      single paragraph spanning two topics becomes two topic-coherent chunks. The
      character split is only a last-resort failsafe (embeddings unavailable, or a
      single sentence over budget). See _chunk_hybrid / _semantic_split.
    """
    strategy = (config or {}).get("chunking", {}).get("strategy", "fixed")
    if strategy == "semantic":
        return _chunk_hybrid(text, max_tokens, config or {})
    return _chunk_fixed(text, max_tokens)


def _chunk_fixed(text: str, max_tokens: int) -> list[dict[str, Any]]:
    """Paragraph-merge chunking; oversized paragraphs are char-hard-split."""
    paragraphs: list[str] = []
    for p in (p.strip() for p in re.split(r"\n{2,}", text) if p.strip()):
        if _estimate_tokens(p) <= max_tokens:
            paragraphs.append(p)
        else:
            paragraphs.extend(_char_split(p, max_tokens))
    chunks: list[dict[str, Any]] = []
    current_parts: list[str] = []
    current_tokens = 0
    current_heading = ""

    for para in paragraphs:
        heading = _detect_heading(para)
        if heading:
            current_heading = heading
        para_tokens = _estimate_tokens(para)
        if current_tokens + para_tokens > max_tokens and current_parts:
            chunks.append({
                "text": "\n\n".join(current_parts),
                "section_heading": current_heading,
            })
            current_parts = []
            current_tokens = 0
        current_parts.append(para)
        current_tokens += para_tokens

    if current_parts:
        chunks.append({
            "text": "\n\n".join(current_parts),
            "section_heading": current_heading,
        })

    # Fallback: if text produced no paragraphs, treat whole text as one chunk
    if not chunks and text.strip():
        chunks.append({"text": text.strip(), "section_heading": ""})

    return chunks


def _chunk_hybrid(text: str, max_tokens: int, config: dict[str, Any]) -> list[dict[str, Any]]:
    """Structural partition (paragraph/heading) + semantic split of WIDE paragraphs.

    Small paragraphs merge to max_tokens as in _chunk_fixed (cheap, no embeddings).
    A wide paragraph — one whose size reaches semantic_min_tokens, so it may pack
    several key concepts — is analysed with sentence embeddings and split at topic
    boundaries; its parts are emitted directly (NOT re-fed to the merge loop, so the
    semantic boundaries survive). semantic_min_tokens sits well below max_tokens, so
    embeddings drive the boundaries of substantial text rather than only rescuing
    oversized glued blobs.
    """
    chunk_cfg = config.get("chunking", {})
    semantic_min = chunk_cfg.get("semantic_min_tokens") or max(1, max_tokens // 2)

    chunks: list[dict[str, Any]] = []
    current_parts: list[str] = []
    current_tokens = 0
    current_heading = ""

    def _flush() -> None:
        nonlocal current_parts, current_tokens
        if current_parts:
            chunks.append({
                "text": "\n\n".join(current_parts),
                "section_heading": current_heading,
            })
            current_parts = []
            current_tokens = 0

    for para in (p.strip() for p in re.split(r"\n{2,}", text) if p.strip()):
        heading = _detect_heading(para)
        if heading:
            current_heading = heading
        para_tokens = _estimate_tokens(para)
        if para_tokens < semantic_min:
            # Small paragraph: structural merge (author's own blank-line boundary).
            if current_tokens + para_tokens > max_tokens and current_parts:
                _flush()
            current_parts.append(para)
            current_tokens += para_tokens
        else:
            # Wide (multi-concept candidate) or oversized: flush the buffer, then
            # split at semantic sentence boundaries and emit each part directly.
            _flush()
            for part in _semantic_split(para, max_tokens, config):
                chunks.append({"text": part, "section_heading": current_heading})

    _flush()
    if not chunks and text.strip():
        chunks.append({"text": text.strip(), "section_heading": ""})
    return chunks


def _split_sentences(text: str) -> list[str]:
    """Split on sentence-final punctuation or newlines. Best-effort, dependency-free."""
    parts = re.split(r"(?<=[.!?])\s+|\n+", text)
    return [s.strip() for s in parts if s.strip()]


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(y * y for y in b) ** 0.5
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def _breakpoint_threshold(dists: list[float], zscore: float) -> float:
    """A topic shift is an OUTLIER adjacent-sentence distance: threshold at
    mean + zscore*stddev. Robust for the few-sentence case (a percentile there
    just selects the max, so nothing ever exceeds it). std==0 → threshold==mean,
    and the strict `>` comparison then yields no semantic breaks (no outliers)."""
    if not dists:
        return float("inf")
    mean = sum(dists) / len(dists)
    std = (sum((d - mean) ** 2 for d in dists) / len(dists)) ** 0.5
    return mean + zscore * std


def _semantic_split(para: str, max_tokens: int, config: dict[str, Any]) -> list[str]:
    """Split a wide paragraph at semantic sentence boundaries.

    Groups consecutive sentences up to max_tokens, forcing an extra break where
    the cosine distance between adjacent sentences is an outlier (mean+z*std) — so
    a paragraph carrying two concepts becomes two chunks even when it fits under
    max_tokens. Falls back to _char_split when there is nothing to embed (single
    sentence) or the embedder is unavailable.
    """
    # Cap every unit at max_tokens BEFORE embedding: a run-on "sentence" (scraped
    # HTML with no . ! ? or newline can be thousands of tokens) would otherwise
    # blow past the embedder's context window and make Ollama's /api/embeddings
    # return 500. Chunks of <= max_tokens embed reliably (the storage path proves
    # it), so char-split any monster sentence into embed-safe units first.
    units: list[str] = []
    for sentence in _split_sentences(para):
        if _estimate_tokens(sentence) > max_tokens:
            units.extend(_char_split(sentence, max_tokens))
        else:
            units.append(sentence)
    if len(units) <= 1:
        return _char_split(para, max_tokens)
    try:
        vectors = embed_many(units, config)
    except Exception as exc:
        logger.warning("Semantic split: embedding failed (%s) — char-splitting", exc)
        return _char_split(para, max_tokens)

    dists = [1.0 - _cosine(vectors[i], vectors[i + 1]) for i in range(len(vectors) - 1)]
    zscore = config.get("chunking", {}).get("semantic_breakpoint_zscore", 1.0)
    threshold = _breakpoint_threshold(dists, zscore)

    groups: list[str] = []
    current: list[str] = []
    current_tokens = 0
    for i, unit in enumerate(units):
        ut = _estimate_tokens(unit)
        would_exceed = current and current_tokens + ut > max_tokens
        topic_shift = i > 0 and current and dists[i - 1] > threshold
        if would_exceed or topic_shift:
            groups.append(" ".join(current))
            current, current_tokens = [], 0
        current.append(unit)
        current_tokens += ut
    if current:
        groups.append(" ".join(current))
    return groups or _char_split(para, max_tokens)


# ── URL fetch ────────────────────────────────────────────────────────────────

def _rewrite_arxiv_url(url: str) -> str:
    """Rewrite arxiv abstract URLs to direct PDF download URLs."""
    # https://arxiv.org/abs/1706.03762 → https://arxiv.org/pdf/1706.03762
    return re.sub(r"arxiv\.org/abs/(\S+)", r"arxiv.org/pdf/\1", url)


def _derive_filename(url: str, response: httpx.Response) -> str:
    """Derive a safe local filename from Content-Disposition header or URL path."""
    cd = response.headers.get("content-disposition", "")
    match = re.search(r'filename="?([^";\s]+)"?', cd)
    if match:
        name = match.group(1)
    else:
        name = Path(urlparse(url).path).name or "document"
    # Sanitise: keep only safe characters
    name = re.sub(r"[^\w.\-]", "_", name)
    if not name:
        name = "document"
    # Ensure a recognised extension; default to .html for web pages
    if Path(name).suffix.lower() not in (".pdf", ".html", ".md", ".txt"):
        ct = response.headers.get("content-type", "")
        if "pdf" in ct:
            name += ".pdf"
        else:
            name += ".html"
    return name


def fetch_and_ingest(
    url: str,
    vault_root: str,
    db_path: str,
    config: dict[str, Any],
    project: str = "default",
    force: bool = False,
) -> dict[str, Any]:
    """Fetch a URL, save to vault/{project}/raw/, and ingest the downloaded file."""
    fetch_cfg = config.get("fetch", {})
    timeout = fetch_cfg.get("timeout_seconds", 30)
    user_agent = fetch_cfg.get("user_agent", "pkms/0.1")
    arxiv_rewrite = fetch_cfg.get("arxiv_rewrite", True)

    effective_url = _rewrite_arxiv_url(url) if arxiv_rewrite else url

    logger.info("Fetching %s …", effective_url)

    def _fetch():
        r = httpx.get(
            effective_url,
            timeout=timeout,
            headers={"User-Agent": user_agent},
            follow_redirects=True,
        )
        r.raise_for_status()
        return r

    # Shared retry policy (pkms.llm.retry_transient): transient-only retries,
    # Retry-After honoured on 429/503 (arxiv/web rate-limit downloads).
    try:
        response = retry_transient(_fetch, retries=3, label=f"Fetch {url}")
    except Exception as exc:
        logger.error("Fetch failed for %s: %s", url, exc)
        raise

    validate_project(project)
    filename = _derive_filename(effective_url, response)
    # vault_root is the repo root; the vault directory lives at vault_root/vault/
    vault_dir = Path(vault_root) / "vault"
    raw_dir = vault_dir / project / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    dest_abs = raw_dir / filename
    vault_rel = str(Path("vault") / project / "raw" / filename)

    guard_write("ingestor", str(dest_abs), str(vault_dir))

    # Drop the cross-process ingest marker BEFORE the file lands on disk, so the
    # watcher (which fires on the file appearing) always sees it and defers.
    # Marking only after fetch+ingest (as the coordinator used to) left a window
    # where the watcher ingested the same file in parallel → double embed + a
    # wiki-lock timeout. The coordinator clears this marker in its finally once
    # the whole ingest+compile cycle is done; on failure here we clear it
    # ourselves so a failed URL ingest doesn't block the watcher forever.
    mark_ingesting(vault_rel)
    try:
        dest_abs.write_bytes(response.content)
        logger.info("Fetched %s → %s", url, dest_abs)
        return ingest(vault_rel, vault_root, db_path, config, project=project, force=force)
    except Exception:
        clear_ingesting(vault_rel)
        raise


# ── main ingest function ──────────────────────────────────────────────────────

def ingest(
    path: str,
    vault_root: str,
    db_path: str,
    config: dict[str, Any],
    project: str = "default",
    force: bool = False,
) -> dict[str, Any]:
    """Ingest a single file into Qdrant raw collection, tagged with its project.

    Returns a result dict with status and indexing metadata. The caller
    (Coordinator) is responsible for writing the result to .search-index.

    force=True re-embeds even when the byte/content hash is unchanged — used to
    backfill after a payload-schema change (unchanged files would otherwise skip).
    """
    validate_project(project)
    abs_path = str(Path(vault_root) / path) if not Path(path).is_absolute() else path

    if not Path(abs_path).exists():
        raise FileNotFoundError(f"Ingest target not found: {abs_path}")

    file_hash = _hash_file(abs_path)
    collection = config["qdrant"]["collections"]["raw"]
    existing = get_file(db_path, path)

    # Fast path: identical raw bytes → nothing to do (static files, e.g. arXiv PDFs).
    if not force and existing and existing["hash"] == file_hash:
        logger.info("Skipping %s — already indexed at current byte hash", path)
        return {"status": "SKIPPED", "path": path, "hash": file_hash}

    # Parse (needed for the content check AND for chunking).
    logger.info("Parsing %s …", path)
    try:
        text = parse(abs_path)
    except Exception as exc:
        logger.error("Parse failed for %s: %s", path, exc)
        raise
    logger.info("Parsed %d characters", len(text))

    # Content path: bytes drifted (dynamic HTML injects per-request tokens) but the
    # EXTRACTED text is unchanged → skip the expensive re-embed/re-compile. The old
    # Qdrant points are still valid, so leave them in place.
    content_hash = _hash_text(text)
    if (not force and existing and existing.get("content_hash")
            and existing["content_hash"] == content_hash):
        logger.info("Skipping %s — content unchanged (raw bytes drifted only)", path)
        return {"status": "SKIPPED", "path": path, "hash": file_hash}

    # Delete old Qdrant points if re-indexing
    if existing and existing["qdrant_ids"]:
        try:
            delete_by_ids(collection, existing["qdrant_ids"], config)
        except Exception as exc:
            logger.warning("Could not delete old Qdrant points for %s: %s", path, exc)

    # Metadata extraction
    logger.info("Extracting metadata (LLM) …")
    meta = extract_metadata(text, path, config)
    logger.info("Metadata: %s — tags %s", meta["title"], meta["tags"])

    # Chunk
    max_tokens = config["chunking"]["max_tokens"]
    chunks = chunk(text, max_tokens, config)
    chunk_total = len(chunks)
    logger.info("Chunked into %d chunks (max %d tokens each)", chunk_total, max_tokens)

    # Embed (batched /api/embed) + one batched upsert for the document; the
    # payload schema (incl. the grounding-critical "text" field) lives in
    # pkms.indexing, shared with the compiler's wiki indexing.
    timestamp = datetime.now(timezone.utc).isoformat()
    qdrant_ids = embed_and_upsert_chunks(
        collection, path, chunks,
        {
            "project": project,
            "title": meta["title"],
            "summary_1line": meta["summary_1line"],
            "tags": meta["tags"],
            "doc_type": meta["doc_type"],
            "hash": file_hash,
            "agent": "ingestor",
        },
        config,
        timestamp=timestamp,
        log_progress=True,
    )

    logger.info("Ingested %s — %d chunks, tags: %s", path, chunk_total, meta["tags"])

    return {
        "status": "DONE",
        "path": path,
        "project": project,
        "hash": file_hash,
        "content_hash": content_hash,
        "qdrant_ids": qdrant_ids,
        "collection": collection,
        "title": meta["title"],
        "tags": meta["tags"],
        "n_chunks": chunk_total,
        "indexed_at": timestamp,
    }
