import hashlib
import json
import logging
import re
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import git

from pkms.db import (
    get_article_sources,
    get_stale_articles,
    upsert_article_source,
    upsert_file,
)
from pkms.embed import embed
from pkms.guards import guard_write, validate_project
from pkms.ingestor import chunk, _estimate_tokens
from pkms.llm import complete
from pkms.indexing import embed_and_upsert_chunks
from pkms.qdrant_store import delete_by_ids, scroll, search

logger = logging.getLogger(__name__)

# ── helpers ───────────────────────────────────────────────────────────────────

def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _hash_text(text: str) -> str:
    return "sha256:" + hashlib.sha256(text.encode()).hexdigest()


def _strip_code_fence(text: str) -> str:
    """Strip an outer ```/```markdown fence the LLM sometimes wraps output in.

    Only removes a fence at the very start and end of the whole response, so
    code fences *inside* the article are preserved. A leading fence here pushes
    the YAML frontmatter off line 1, which breaks frontmatter parsing in the
    wiki browser and the _index/crosslink passes.
    """
    text = text.strip()
    text = re.sub(r"^```[^\n]*\n?", "", text)
    text = re.sub(r"\n?```$", "", text)
    return text.strip()


def _slug(raw_path: str) -> str:
    return Path(raw_path).stem.lower().replace(" ", "_")


def _project_of(path: str) -> str:
    """Extract the project from a vault-relative path like vault/{project}/raw/x.pdf.

    Raises ValueError on any non-vault-relative path (e.g. an absolute path).
    Previously this silently returned "default", which let an absolute path from
    the upload boundary misplace compiled articles into vault/default/ while the
    DB recorded the correct project.
    """
    parts = Path(path).parts
    if len(parts) >= 3 and parts[0] == "vault":
        return parts[1]
    raise ValueError(
        f"Cannot determine project from non-vault-relative path {path!r}; "
        "paths must be vault-relative (vault/{project}/...)."
    )


def _load_conventions(vault_dir: Path, project: str) -> str:
    """Return the project's writing schema from vault/{project}/CLAUDE.md, or "".

    This is the human-editable "schema" doc (à la Karpathy's CLAUDE.md/AGENTS.md):
    team-authored conventions that steer how the LLM writes this project's wiki.
    Absent file → "" → the compiler's built-in prompts are used verbatim (no
    behaviour change).
    """
    schema_path = vault_dir / project / "CLAUDE.md"
    if schema_path.exists():
        return schema_path.read_text(encoding="utf-8").strip()
    return ""


def _wiki_path(raw_path: str, project: str | None = None) -> str:
    parsed = _project_of(raw_path)
    if project is not None and parsed != project:
        raise ValueError(
            f"Project mismatch for {raw_path!r}: path resolves to {parsed!r} "
            f"but compile was invoked for project {project!r}."
        )
    return f"vault/{parsed}/wiki/articles/{_slug(raw_path)}.md"


def _get_uncompiled_raw_paths(db_path: str, project: str) -> list[str]:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    rows = conn.execute("""
        SELECT path FROM files
        WHERE collection = 'raw'
          AND project = ?
          AND path NOT IN (SELECT DISTINCT raw_path FROM article_sources)
    """, (project,)).fetchall()
    conn.close()
    return [r["path"] for r in rows]


# ── discover work ─────────────────────────────────────────────────────────────

def _discover_work(
    scope: dict[str, Any],
    db_path: str,
    config: dict[str, Any],
    project: str,
) -> list[dict[str, Any]]:
    """Return list of {wiki_path, sources: [{raw_path, hash}]} to compile."""
    work: dict[str, dict[str, Any]] = {}

    # Stale articles (source changed since last compile)
    for row in get_stale_articles(db_path, project=project):
        wp = row["wiki_path"]
        if wp not in work:
            work[wp] = {"wiki_path": wp, "sources": []}
        work[wp]["sources"].append({"raw_path": row["raw_path"], "hash": row["current_hash"]})

    # New sources never compiled yet — path AND hash in one query (a previous
    # version re-opened a connection per path inside this loop).
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    rows = conn.execute("""
        SELECT path, hash FROM files
        WHERE collection = 'raw'
          AND project = ?
          AND path NOT IN (SELECT DISTINCT raw_path FROM article_sources)
    """, (project,)).fetchall()
    conn.close()
    for row in rows:
        wp = _wiki_path(row["path"], project)
        if wp not in work:
            work[wp] = {"wiki_path": wp, "sources": []}
        work[wp]["sources"].append({"raw_path": row["path"], "hash": row["hash"]})

    items = list(work.values())

    # Filter by scope
    scope_type = scope.get("type", "full")
    if scope_type == "doc":
        source_filter = scope["source"]
        items = [i for i in items if any(s["raw_path"] == source_filter for s in i["sources"])]
    elif scope_type == "topic":
        topic = scope["topic"].lower()
        collection = config["qdrant"]["collections"]["raw"]
        filtered = []
        for item in items:
            matched = False
            for src in item["sources"]:
                # Fast path: topic string in file path
                if topic in src["raw_path"].lower():
                    matched = True
                    break
                # Semantic fallback: check tags stored in Qdrant raw chunk payloads
                try:
                    points = scroll(collection, src["raw_path"], config)
                    chunk_tags = [
                        t.lower()
                        for p in points
                        for t in p["payload"].get("tags", [])
                    ]
                    if any(topic in t for t in chunk_tags):
                        matched = True
                        break
                except Exception as exc:
                    logger.warning("Tag lookup failed for %s: %s", src["raw_path"], exc)
            if matched:
                filtered.append(item)
        items = filtered

    return items


# ── LLM prompts ───────────────────────────────────────────────────────────────

_WRITE_PROMPT = """\
You are a technical wiki writer. Write a comprehensive wiki article in Markdown.

Source material (chunks from raw documents):
{chunks_text}

Requirements:
- Start with YAML frontmatter: title, tags (list), sources (list of raw paths), date
- Write a 1-2 sentence summary paragraph immediately after frontmatter
- Organise into clearly named sections with ## headings
- Use [[wiki-link]] syntax where you reference related concepts (placeholders, filled later)
- End with a ## Sources section listing the raw document paths
- Write in clear, precise technical prose. No padding.

Reply with the full Markdown article only — no explanation outside the article.
"""

_UPDATE_PROMPT = """\
You are a technical wiki editor. Update the existing wiki article below with new source material.
Update minimally: preserve structure and existing content, add or correct only what the new material changes.

Existing article:
{existing}

New/changed source material:
{chunks_text}

Reply with the complete updated Markdown article only.
"""

# B2 hierarchical compile — map each oversized group of source chunks into dense
# faithful notes, then reduce all notes into the final article.
_MAP_PROMPT = """\
Extract the key factual content from these source excerpts as dense, faithful notes.
Preserve technical terms, method names, datasets, numbers, and results verbatim.
Do NOT invent anything not present in the excerpts. No preamble, notes only.

Source excerpts:
{chunks_text}
"""

_REDUCE_PROMPT = """\
You are a technical wiki writer. Write a comprehensive wiki article in Markdown from the
extracted notes below (the notes are faithful digests of the source document, in order).

Notes:
{chunks_text}

Requirements:
- Start with YAML frontmatter: title, tags (list), sources (list of raw paths), date
- Write a 1-2 sentence summary paragraph immediately after frontmatter
- Organise into clearly named sections with ## headings
- Use [[wiki-link]] syntax where you reference related concepts (placeholders, filled later)
- End with a ## Sources section listing the raw document paths
- Ground strictly in the notes. No padding, no invention.

Reply with the full Markdown article only — no explanation outside the article.
"""

# B9 output-side chunking — write arbitrarily long articles section by section so
# no single generation hits the output-token cap (works on free-tier / local models).
_OUTLINE_PROMPT = """\
You are planning a technical wiki article from the source material below.
Reply with a JSON object ONLY:
{{"frontmatter": {{"title": "…", "summary_1line": "one sentence", "tags": ["…", "…"]}},
  "sections": ["Introduction", "Architecture", "Results", "…"]}}
Choose 3–8 section headings that comprehensively organise the material, in reading
order. Do NOT write any section bodies here. Ground title/summary/tags in the material.

Source material:
{source}
"""

_SECTION_PROMPT = """\
You are writing ONE section of a technical wiki article titled "{title}".
Write ONLY the "## {section}" section — its heading and body, nothing else: no
frontmatter, no other sections, no Sources list. Use [[wiki-link]] syntax for related
concepts (placeholders, filled later). Ground strictly in the source material below;
no invention, no padding.

Full article outline (context — do NOT write the other sections):
{outline}

Source material:
{source}
"""

_CROSSLINK_PROMPT = """\
Given these new wiki articles and related existing articles, suggest [[wiki-links]] to add.
Only suggest high-confidence links (similarity >= {threshold}).

New articles: {new_slugs}
Related articles found: {related}

Reply with a JSON array of objects: [{{"source_slug": "...", "target_slug": "...", "anchor_text": "..."}}]
Reply with [] if no confident suggestions.
"""


# ── article compilation ───────────────────────────────────────────────────────

_CHUNK_SEP = "\n\n---\n\n"


def _chunk_texts(sources: list[dict[str, Any]], config: dict[str, Any]) -> list[str]:
    """Sorted per-chunk texts (section heading + body) for one article's sources."""
    collection = config["qdrant"]["collections"]["raw"]
    all_chunks: list[dict[str, Any]] = []
    for src in sources:
        all_chunks.extend(scroll(collection, src["raw_path"], config))
    all_chunks.sort(key=lambda p: (
        p["payload"].get("path", ""),
        p["payload"].get("chunk_index", 0),
    ))
    out: list[str] = []
    for p in all_chunks:
        heading = p["payload"].get("section_heading", "")
        text = p["payload"].get("text", "")
        out.append(f"{heading}\n\n{text}" if heading else text)
    return out


def _group_by_budget(units: list[str], budget: int) -> list[list[str]]:
    """Greedily pack text units into groups whose est. tokens each stay <= budget.
    A unit already over budget becomes its own group (chunks are size-capped at
    ingest, so this is rare)."""
    groups: list[list[str]] = []
    cur: list[str] = []
    cur_tok = 0
    for u in units:
        ut = _estimate_tokens(u)
        if cur and cur_tok + ut > budget:
            groups.append(cur)
            cur, cur_tok = [], 0
        cur.append(u)
        cur_tok += ut
    if cur:
        groups.append(cur)
    return groups


def _mapreduce_notes(units: list[str], config: dict[str, Any]) -> str:
    """Collapse source text units into a notes blob that fits max_prompt_tokens, by
    repeatedly map-summarising over-budget groups. Handles arbitrarily large docs
    (recurses until the joined notes fit, or gives up gracefully)."""
    budget = config.get("compile", {}).get("max_prompt_tokens", 3500)
    map_out = config.get("compile", {}).get("map_num_predict", 512)
    guard = 0
    while _estimate_tokens(_CHUNK_SEP.join(units)) > budget and len(units) > 1:
        guard += 1
        groups = _group_by_budget(units, budget)
        if len(groups) == len(units) or guard > 10:
            # No progress (each unit already ~budget) or runaway — stop; the reduce
            # step will get a slightly-over-budget notes blob, still far under the
            # single-pass size that triggered this.
            logger.warning("Map-reduce: stopping collapse at %d units (pass %d)", len(units), guard)
            break
        logger.info("Map-reduce: %d units → %d groups (pass %d)", len(units), len(groups), guard)
        units = [
            _strip_code_fence(complete("compiler",
                                       _MAP_PROMPT.format(chunks_text=_CHUNK_SEP.join(g)),
                                       config, num_predict=map_out))
            for g in groups
        ]
    return _CHUNK_SEP.join(units)


def _looks_truncated(article: str) -> bool:
    """The WRITE/UPDATE/REDUCE prompts all end the article with a Sources section;
    its absence means generation was cut off before finishing (hit the output cap)."""
    return "## Sources" not in article and "##Sources" not in article


def _prepend_conventions(prompt: str, conventions: str) -> str:
    """Prepend the project schema AFTER .format() so braces in it can't break it."""
    if not conventions:
        return prompt
    return f"Project writing conventions (follow these where they apply):\n{conventions}\n\n---\n\n{prompt}"


def _sources_section(sources: list[dict[str, Any]]) -> str:
    paths = sorted({s["raw_path"] for s in sources})
    return "## Sources\n\n" + "\n".join(f"- {p}" for p in paths) + "\n"


def _assemble_sectioned(fm: dict[str, Any], section_bodies: list[str],
                        sources: list[dict[str, Any]], fallback_title: str) -> str:
    """Build the final article: frontmatter + summary + sections + deterministic Sources."""
    title = str(fm.get("title") or fallback_title)
    summary = str(fm.get("summary_1line") or "")
    tags = fm.get("tags") if isinstance(fm.get("tags"), list) else []
    src_paths = sorted({s["raw_path"] for s in sources})
    lines = [
        "---",
        f'title: "{title}"',
        f"tags: [{', '.join(str(t) for t in tags)}]",
        f"sources: [{', '.join(src_paths)}]",
        f"date: {_now()[:10]}",
        f'summary_1line: "{summary}"',
        "---",
        "",
    ]
    if summary:
        lines += [summary, ""]
    for body in section_bodies:
        lines += [body.strip(), ""]
    return "\n".join(lines) + _sources_section(sources)


def _compile_sectioned(source_text: str, sources: list[dict[str, Any]],
                       conventions: str, config: dict[str, Any], fallback_title: str) -> str | None:
    """Write the article SECTION BY SECTION (B9): outline first, then one bounded
    call per section, so no single generation hits the output-token cap. Returns
    None if the outline can't be parsed (caller falls back to single-pass)."""
    outline_prompt = _prepend_conventions(_OUTLINE_PROMPT.format(source=source_text), conventions)
    try:
        raw = _strip_code_fence(complete("compiler", outline_prompt, config,
                                         num_predict=1024, response_format="json"))
        outline = json.loads(raw)
    except Exception as exc:
        logger.warning("Sectioned compile: outline failed (%s) — falling back", exc)
        return None
    fm = outline.get("frontmatter") if isinstance(outline.get("frontmatter"), dict) else {}
    sections = [s for s in (outline.get("sections") or []) if isinstance(s, str) and s.strip()
                and "source" not in s.lower()]   # Sources is appended deterministically
    if not sections:
        logger.warning("Sectioned compile: outline had no usable sections — falling back")
        return None

    title = str(fm.get("title") or fallback_title)
    outline_md = "\n".join(f"- {s}" for s in sections)
    section_out = config.get("compile", {}).get("section_num_predict", 2048)
    bodies: list[str] = []
    for heading in sections:
        sp = _prepend_conventions(_SECTION_PROMPT.format(
            title=title, section=heading, outline=outline_md, source=source_text), conventions)
        body = _strip_code_fence(complete("compiler", sp, config, num_predict=section_out)).strip()
        if not body:
            continue
        bodies.append(body if body.lstrip().startswith("#") else f"## {heading}\n\n{body}")
    if not bodies:
        logger.warning("Sectioned compile: no section bodies produced — falling back")
        return None
    logger.info("Sectioned compile: assembled %d sections", len(bodies))
    return _assemble_sectioned(fm, bodies, sources, fallback_title)


def _compile_article(
    wiki_path: str,
    sources: list[dict[str, Any]],
    vault_dir: Path,
    config: dict[str, Any],
    conventions: str = "",
) -> str:
    """Retrieve chunks, call the LLM, return article Markdown.

    Large sources are handled on both sides of the LLM: the INPUT is bounded by B2
    map-reduce (chunks → faithful notes ≤ max_prompt_tokens), and the OUTPUT is
    bounded by B9 section-by-section generation (outline, then one bounded call per
    section) so an arbitrarily long article never hits the output-token cap — even
    on free-tier / local models. Single-pass is kept for the common short case, with
    a sectioned rescue if it truncates.

    conventions: optional project schema (vault/{project}/CLAUDE.md).
    """
    units = _chunk_texts(sources, config)
    chunks_text = _CHUNK_SEP.join(units)
    budget = config.get("compile", {}).get("max_prompt_tokens", 3500)
    over_budget = _estimate_tokens(chunks_text) > budget
    article_file = vault_dir / Path(wiki_path).relative_to("vault")
    existing = article_file.read_text(encoding="utf-8") if article_file.exists() else None
    sectioned_enabled = config.get("compile", {}).get("sectioned_compile", True)
    fallback_title = Path(wiki_path).stem

    # Source context: bounded notes for large docs (B2), raw chunks otherwise.
    if over_budget:
        logger.info("Compile: %s over budget (~%d tok) — map-reduce notes",
                    fallback_title, _estimate_tokens(chunks_text))
        source_text = _mapreduce_notes(units, config)
    else:
        source_text = chunks_text

    # Large source ⇒ article likely exceeds the output cap ⇒ write it section by
    # section from the start (no wasted truncated single-pass, and each call stays
    # small enough for free-tier TPM limits).
    if sectioned_enabled and over_budget:
        art = _compile_sectioned(source_text, sources, conventions, config, fallback_title)
        if art:
            return art
        logger.warning("Compile: %s sectioned failed — trying single-pass", fallback_title)

    # Single-pass (the common, short case).
    if over_budget:
        prompt = _REDUCE_PROMPT.format(chunks_text=source_text)
    elif existing:
        prompt = _UPDATE_PROMPT.format(existing=existing, chunks_text=source_text)
    else:
        prompt = _WRITE_PROMPT.format(chunks_text=source_text)
    prompt = _prepend_conventions(prompt, conventions)

    article_out = config.get("compile", {}).get("article_num_predict", 4096)
    article = _strip_code_fence(complete("compiler", prompt, config, num_predict=article_out))

    # Rescue a truncated single-pass by rebuilding section by section.
    if _looks_truncated(article) and sectioned_enabled:
        logger.info("Compile: %s single-pass truncated — rebuilding section by section", fallback_title)
        art = _compile_sectioned(source_text, sources, conventions, config, fallback_title)
        if art:
            return art
    if _looks_truncated(article):
        logger.warning("Compile: %s looks truncated (no Sources section — hit the "
                       "article_num_predict=%d output cap)", fallback_title, article_out)
    return article


def _embed_and_upsert_wiki(
    wiki_path: str,
    article_md: str,
    config: dict[str, Any],
    project: str,
) -> list[str]:
    """Chunk, embed, and upsert wiki article. Returns list of point IDs.

    The payload schema (incl. the grounding-critical "text" field) lives in
    pkms.indexing, shared with the ingestor's raw-document indexing.
    """
    collection = config["qdrant"]["collections"]["wiki"]
    max_tokens = config["chunking"]["max_tokens"]
    chunks = chunk(article_md, max_tokens, config)
    title = _extract_frontmatter_field(article_md, "title") or Path(wiki_path).stem

    # Delete existing wiki points for this path before re-upserting
    existing_points = scroll(collection, wiki_path, config)
    if existing_points:
        delete_by_ids(collection, [p["id"] for p in existing_points], config)

    return embed_and_upsert_chunks(
        collection, wiki_path, chunks,
        {
            "project": project,
            "title": title,
            "summary_1line": "",
            "tags": _extract_frontmatter_tags(article_md),
            "hash": _hash_text(article_md),
            "agent": "compiler",
        },
        config,
        timestamp=_now(),
    )


def _extract_frontmatter_field(md: str, field: str) -> str:
    match = re.search(rf"^{field}:\s*[\"']?(.+?)[\"']?\s*$", md, re.MULTILINE)
    return match.group(1).strip() if match else ""


def _extract_frontmatter_tags(md: str) -> list[str]:
    # Inline form: tags: [a, b, c]
    match = re.search(r"^tags:\s*\[(.+?)\]", md, re.MULTILINE)
    if match:
        return [t.strip().strip('"\'') for t in match.group(1).split(",") if t.strip()]
    # Block form:
    #   tags:
    #     - a
    #     - b
    m = re.search(r"^tags:\s*$", md, re.MULTILINE)
    if not m:
        return []
    tags: list[str] = []
    for line in md[m.end():].splitlines()[1:]:
        if re.match(r"\s*-\s+", line):
            tags.append(re.sub(r"\s*-\s+", "", line, count=1).strip().strip('"\''))
        elif line.strip() == "":
            continue
        else:
            break  # next frontmatter key (e.g. sources:) ends the list
    return tags


# ── _index.md ─────────────────────────────────────────────────────────────────

def _rebuild_index(vault_dir: Path, project: str) -> None:
    """Regenerate wiki/_index.md: one `- [[slug]] — summary` line per article.

    Pure string formatting over data already in hand (slug + frontmatter
    summary) — deliberately NOT an LLM call: an earlier version paid one
    completion per compile cycle (plus its 429 exposure and nondeterministic
    output) just to reformat this manifest.
    """
    articles_dir = vault_dir / project / "wiki" / "articles"
    if not articles_dir.exists():
        return
    lines = []
    for md_file in sorted(articles_dir.glob("*.md")):
        slug = md_file.stem
        content = md_file.read_text(encoding="utf-8")
        summary = _extract_frontmatter_field(content, "summary_1line") or _extract_frontmatter_field(content, "title")
        lines.append(f"- [[{slug}]] — {summary}")

    if not lines:
        return

    index_path = vault_dir / project / "wiki" / "_index.md"
    guard_write("compiler", str(index_path), str(vault_dir))
    index_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ── cross-link pass ───────────────────────────────────────────────────────────

def _crosslink_pass(
    new_wiki_paths: list[str],
    vault_dir: Path,
    config: dict[str, Any],
    project: str,
) -> int:
    threshold = config["compile"]["crosslink_threshold"]
    wiki_collection = config["qdrant"]["collections"]["wiki"]
    links_added = 0

    for wiki_path in new_wiki_paths:
        article_file = vault_dir / Path(wiki_path).relative_to("vault")
        if not article_file.exists():
            continue
        article_text = article_file.read_text(encoding="utf-8")
        title = _extract_frontmatter_field(article_text, "title") or article_file.stem
        try:
            vec = embed(title, config)
            related = search(wiki_collection, vec, top_k=5, config=config, project=project)
        except Exception as exc:
            logger.warning("Crosslink search failed for %s: %s", wiki_path, exc)
            continue

        related_slugs = [
            Path(r["payload"].get("path", "")).stem
            for r in related
            if r["score"] >= threshold and r["payload"].get("path") != wiki_path
        ]
        if not related_slugs:
            continue

        prompt = _CROSSLINK_PROMPT.format(
            threshold=threshold,
            new_slugs=[Path(p).stem for p in new_wiki_paths],
            related=related_slugs,
        )
        try:
            raw = complete("compiler", prompt, config)
            raw = re.sub(r"^```[^\n]*\n?", "", raw.strip())
            raw = re.sub(r"```$", "", raw.strip())
            suggestions = json.loads(raw) if raw.strip().startswith("[") else []
        except Exception as exc:
            logger.warning("Crosslink LLM call failed: %s", exc)
            continue

        new_links: list[str] = []
        for suggestion in suggestions:
            target = suggestion.get("target_slug", "")
            anchor = suggestion.get("anchor_text", target)
            if not target or f"[[{target}]]" in article_text:
                continue
            link = f"[[{target}|{anchor}]]" if anchor != target else f"[[{target}]]"
            new_links.append(f"- {link}")

        if new_links:
            # Append or extend a "## See also" section at the end of the article
            see_also_header = "## See also"
            if see_also_header in article_text:
                article_text = article_text.rstrip() + "\n" + "\n".join(new_links) + "\n"
            else:
                article_text = article_text.rstrip() + f"\n\n{see_also_header}\n\n" + "\n".join(new_links) + "\n"
            guard_write("compiler", str(article_file), str(vault_dir))
            article_file.write_text(article_text, encoding="utf-8")
            links_added += len(new_links)

    return links_added


# ── dangling-link cleanup (after a document is removed) ─────────────────────────

# A [[slug]] or [[slug|anchor]] wiki-link, capturing slug and optional anchor.
_WIKILINK_TOKEN_RE = re.compile(r"\[\[([^\]|]+)(?:\|([^\]]*))?\]\]")


def _strip_dangling_links(
    vault_dir: Path,
    project: str,
    removed_slugs: set[str],
    config: dict[str, Any],
) -> int:
    """Remove [[slug]] references to removed articles from every sibling article.

    - A list item whose link points only at a removed slug is dropped entirely
      (these are the "## See also" entries the crosslink pass appends).
    - An inline [[slug]] elsewhere degrades to its anchor text (or the slug),
      so prose stays readable instead of showing a broken link.
    - A "## See also" heading left with no items is removed.

    Returns the number of article files modified.
    """
    if not removed_slugs:
        return 0
    articles_dir = vault_dir / project / "wiki" / "articles"
    if not articles_dir.exists():
        return 0

    def _targets_removed(text: str) -> bool:
        return any(
            (Path(m.group(1).strip()).stem if "/" in m.group(1) else m.group(1).strip())
            in removed_slugs
            for m in _WIKILINK_TOKEN_RE.finditer(text)
        )

    def _degrade_inline(text: str) -> str:
        def _sub(m: re.Match) -> str:
            raw = m.group(1).strip()
            slug = Path(raw).stem if "/" in raw else raw
            if slug in removed_slugs:
                return (m.group(2) or raw).strip()
            return m.group(0)
        return _WIKILINK_TOKEN_RE.sub(_sub, text)

    modified = 0
    for md_file in sorted(articles_dir.glob("*.md")):
        original = md_file.read_text(encoding="utf-8", errors="replace")
        out_lines: list[str] = []
        for line in original.splitlines():
            is_list_item = re.match(r"^\s*[-*]\s+", line) is not None
            if is_list_item and _targets_removed(line):
                # Drop the whole bullet (a See-also entry pointing at a removed doc)
                continue
            out_lines.append(_degrade_inline(line))

        # Drop a now-empty "## See also" section (header followed by no list items)
        cleaned: list[str] = []
        i = 0
        while i < len(out_lines):
            line = out_lines[i]
            if re.match(r"^#+\s+see also\s*$", line.strip(), re.IGNORECASE):
                j = i + 1
                while j < len(out_lines) and out_lines[j].strip() == "":
                    j += 1
                has_items = j < len(out_lines) and re.match(r"^\s*[-*]\s+", out_lines[j])
                if not has_items:
                    i = j
                    continue
            cleaned.append(line)
            i += 1

        new_text = "\n".join(cleaned)
        if original.endswith("\n") and not new_text.endswith("\n"):
            new_text += "\n"
        if new_text != original:
            guard_write("remover", str(md_file), str(vault_dir))
            md_file.write_text(new_text, encoding="utf-8")
            modified += 1
    return modified


# ── git commit ────────────────────────────────────────────────────────────────

def _git_commit_paths(vault_dir: Path, message: str) -> None:
    """Stage ALL changes under the vault (adds, edits, deletions) and commit.

    Unlike _git_commit (which only `index.add`s given paths), this uses
    `git add -A` so removed files are staged as deletions — needed when a
    document is un-ingested.
    """
    try:
        try:
            repo = git.Repo(str(vault_dir))
        except git.InvalidGitRepositoryError:
            repo = git.Repo.init(str(vault_dir))
            logger.info("Initialised git repository in %s", vault_dir)
        repo.git.add(A=True)
        if not repo.head.is_valid():
            repo.index.commit(message)
        elif repo.index.diff("HEAD") or repo.untracked_files:
            repo.index.commit(message)
    except Exception as exc:
        logger.warning("Git commit (removal) failed: %s", exc)


def _git_commit(vault_dir: Path, rel_paths: list[str], message: str) -> None:
    try:
        try:
            repo = git.Repo(str(vault_dir))
        except git.InvalidGitRepositoryError:
            repo = git.Repo.init(str(vault_dir))
            logger.info("Initialised git repository in %s", vault_dir)
        repo.index.add(rel_paths)
        if not repo.head.is_valid():
            # Fresh repo: HEAD is unborn, diff("HEAD") would raise
            repo.index.commit(message)
        elif repo.index.diff("HEAD"):
            # Commit ONLY when the staged paths differ from HEAD. index.diff("HEAD")
            # already reports newly added files, so the former `or repo.untracked_files`
            # was redundant for its purpose and harmful: it fired an EMPTY commit
            # whenever ANY unrelated untracked file sat in the vault — one per document
            # on a forced reindex of unchanged sources.
            repo.index.commit(message)
    except Exception as exc:
        logger.warning("Git commit failed: %s", exc)


# ── main compile function ─────────────────────────────────────────────────────

def compile(
    scope: dict[str, Any],
    vault_root: str,
    db_path: str,
    lock_token: str,
    config: dict[str, Any],
    project: str = "default",
) -> dict[str, Any]:
    """Compile one project's wiki articles from stale or new raw sources.

    scope: {"type": "full"} | {"type": "doc", "source": "vault/{project}/raw/..."} | {"type": "topic", "topic": "..."}
    lock_token: acquired by Coordinator (scope "wiki:{project}") before spawning this agent.
    Returns result dict; Coordinator is responsible for releasing the lock.
    """
    validate_project(project)
    vault_dir = Path(vault_root) / "vault"
    articles_dir = vault_dir / project / "wiki" / "articles"
    articles_dir.mkdir(parents=True, exist_ok=True)

    work = _discover_work(scope, db_path, config, project)
    if not work:
        logger.info("Compiler: nothing to compile (wiki is current)")
        return {"status": "SKIPPED", "n_articles": 0, "n_links": 0}

    conventions = _load_conventions(vault_dir, project)
    if conventions:
        logger.info("Compiler: applying project schema vault/%s/CLAUDE.md", project)

    n_articles = 0
    new_wiki_paths: list[str] = []
    failed: list[str] = []
    pace = config.get("compile", {}).get("pace_seconds", 0) or 0

    for idx, item in enumerate(work):
        wiki_path = item["wiki_path"]
        sources = item["sources"]

        # Proactive pacing: pause between article compiles to smooth the request
        # burst so a batch trips the LLM rate limit (429) less often. Not before the
        # first, and only when there's more than one to do.
        if pace and idx > 0:
            time.sleep(pace)

        # Compile article via LLM. A single doc that can't compile (persistent rate
        # limit + fallback also failing, or an oversized prompt) must NOT abort the
        # whole batch — log it, skip it, keep going. It stays uncompiled (no
        # article_sources row) so a later compile retries just that one.
        try:
            article_md = _compile_article(wiki_path, sources, vault_dir, config, conventions)
        except Exception as exc:
            logger.error("Compile failed for %s: %s — skipping", wiki_path, exc)
            failed.append(wiki_path)
            continue

        # Write article to filesystem
        article_abs = vault_dir / Path(wiki_path).relative_to("vault")
        article_abs.parent.mkdir(parents=True, exist_ok=True)
        guard_write("compiler", str(article_abs), str(vault_dir))
        article_abs.write_text(article_md, encoding="utf-8")

        # Embed + upsert wiki chunks
        qdrant_ids = _embed_and_upsert_wiki(wiki_path, article_md, config, project)

        # Update .search-index
        article_hash = _hash_text(article_md)
        timestamp = _now()
        db_abs = str(vault_dir / ".search-index")
        guard_write("compiler", db_abs, str(vault_dir))
        upsert_file(
            db_path,
            path=wiki_path,
            hash=article_hash,
            qdrant_ids=qdrant_ids,
            collection="wiki",
            indexed_at=timestamp,
            project=project,
        )
        for src in sources:
            upsert_article_source(
                db_path,
                wiki_path=wiki_path,
                raw_path=src["raw_path"],
                compile_hash=src["hash"],
                compiled_at=timestamp,
            )

        # Git commit
        rel_path = str(Path(wiki_path).relative_to("vault"))
        _git_commit(vault_dir, [rel_path], f"compile: update {Path(wiki_path).stem} from {', '.join(Path(s['raw_path']).name for s in sources)}")

        new_wiki_paths.append(wiki_path)
        n_articles += 1
        logger.info("Compiled %s from %d sources", wiki_path, len(sources))

    # Rebuild _index.md
    try:
        _rebuild_index(vault_dir, project)
    except Exception as exc:
        logger.warning("_index.md rebuild failed (non-fatal): %s", exc)

    # Cross-link pass
    n_links = 0
    try:
        n_links = _crosslink_pass(new_wiki_paths, vault_dir, config, project)
    except Exception as exc:
        logger.warning("Cross-link pass failed (non-fatal): %s", exc)

    if failed:
        logger.warning("Compile: %d article(s) failed and were skipped: %s",
                       len(failed), ", ".join(Path(p).stem for p in failed))
    logger.info("Compile done: %d articles, %d links added, %d failed", n_articles, n_links, len(failed))
    return {"status": "DONE", "n_articles": n_articles, "n_links": n_links,
            "n_failed": len(failed), "failed": failed}
