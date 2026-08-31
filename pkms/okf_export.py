"""OKF (Open Knowledge Format v0.1) export adapter.

Produces an OKF-conformant bundle from a compiled WikiForge wiki WITHOUT touching the
canonical format: it reads vault/<project>/wiki/articles/*.md and emits a PARALLEL
directory whose Markdown files carry OKF's core frontmatter (type, title, description,
resource, tags, timestamp) and whose bodies use standard relative Markdown links
`[text](./slug.md)` instead of `[[wikilink]]`, plus an OKF `index.md`.

Design note (why an adapter, not a format change): the canonical wiki, the linter, the
link-graph UI (pkms/graph.py) and the querier all keep reading the existing frontmatter
and `[[wikilink]]` syntax — nothing is migrated or broken. OKF is a second, opt-in
representation. Only OKF `type` is required by the spec; the rest is best-effort from
what the wiki holds. WikiForge's `sources` (provenance of the *inputs*) has no OKF core
field, so it is preserved as an extension key (OKF allows backward-compatible growth).
Unresolved `[[concept]]` links — which point to no existing file — are downgraded to
plain text, because OKF relative links can only target real files (the canonical wiki
keeps them as the graph's "concept" nodes).

OKF spec: https://cloud.google.com/blog/products/data-analytics/how-the-open-knowledge-format-can-improve-data-sharing
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from pkms.compiler import _extract_frontmatter_field, _extract_frontmatter_tags
from pkms.linter import _extract_wiki_links  # noqa: F401  (shared regex; kept for parity)

_OKF_VERSION = "0.1"
_DEFAULT_TYPE = "Wiki Article"

_FM_RE = re.compile(r"^---\n(.*?)\n---\n?(.*)$", re.DOTALL)
# The target class excludes ] | [ and newline so a MALFORMED source link (e.g. a stray
# `[[Foo>>` closed with the wrong delimiter) cannot greedily swallow the next well-formed
# `[[Bar]]` on the same line — it simply fails to match and is left as literal text.
_LINK_RE = re.compile(r"\[\[([^\]\[|\n]+)(?:\|([^\]\n]*))?\]\]")


def _split_frontmatter(md: str) -> tuple[str, str]:
    """(frontmatter_text_without_delimiters, body). No frontmatter → ('', md)."""
    m = _FM_RE.match(md)
    return (m.group(1), m.group(2)) if m else ("", md)


def _to_timestamp(date_str: str) -> str:
    """Widen a WikiForge `date: YYYY-MM-DD` to an OKF ISO-8601 `timestamp`. An empty
    value stays empty; an already-full timestamp is passed through unchanged."""
    date_str = (date_str or "").strip()
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", date_str):
        return f"{date_str}T00:00:00Z"
    return date_str


def _strip_orphan_fence(body: str) -> str:
    """Drop a stray opening ``` (or ~~~) fence left at the very start of a compiled body
    — a known WikiForge compiler artifact — so the OKF output doesn't render as a broken
    code block. Only when the fence count is ODD (unbalanced): a genuine leading code
    block keeps its closing fence and stays balanced, so it is left untouched."""
    lines = body.lstrip("\n").split("\n")
    if lines and lines[0].strip() in ("```", "~~~"):
        fences = sum(1 for ln in lines if ln.strip().startswith(("```", "~~~")))
        if fences % 2 == 1:
            lines = lines[1:]
    return "\n".join(lines)


def _rewrite_links(body: str, slugs: set[str]) -> str:
    """`[[slug]]` → `[slug](./slug.md)` and `[[slug|anchor]]` → `[anchor](./slug.md)`
    when slug is a real article; an unresolved `[[concept]]` → plain text `concept`."""
    def repl(m: re.Match) -> str:
        target = m.group(1).strip()
        anchor = (m.group(2) or "").strip()
        stem = Path(target).stem if "/" in target else target
        label = anchor or target
        return f"[{label}](./{stem}.md)" if stem in slugs else label
    return _LINK_RE.sub(repl, body)


def _extract_sources(fm_text: str) -> list[str]:
    """WikiForge `sources` (block or inline form) from the raw frontmatter text."""
    mi = re.search(r"^sources:\s*\[(.+?)\]", fm_text, re.MULTILINE)
    if mi:
        return [s.strip().strip("\"'") for s in mi.group(1).split(",") if s.strip()]
    m = re.search(r"^sources:\s*$", fm_text, re.MULTILINE)
    if not m:
        return []
    out: list[str] = []
    for line in fm_text[m.end():].splitlines()[1:]:
        if re.match(r"\s*-\s+", line):
            out.append(re.sub(r"\s*-\s+", "", line, count=1).strip().strip("\"'"))
        elif line.strip() == "":
            continue
        else:
            break
    return out


def _yaml_dq(s: str) -> str:
    """Double-quoted YAML scalar with the minimal escaping we need for titles/descriptions."""
    return '"' + (s or "").replace("\\", "\\\\").replace('"', '\\"') + '"'


def _okf_frontmatter(*, doc_type: str, title: str, description: str, tags: list[str],
                     timestamp: str, resource: str, sources: list[str]) -> str:
    lines = ["---", f"type: {doc_type}"]           # `type` is the only OKF-required field
    if title:
        lines.append(f"title: {_yaml_dq(title)}")
    if description:
        lines.append(f"description: {_yaml_dq(description)}")
    if resource:
        lines.append(f"resource: {resource}")
    if tags:
        lines.append("tags: [" + ", ".join(tags) + "]")
    if timestamp:
        lines.append(f"timestamp: {timestamp}")
    if sources:                                    # WikiForge extension (input provenance)
        lines.append("sources:")
        lines += [f"  - {s}" for s in sources]
    lines.append("---")
    return "\n".join(lines)


def export_okf(vault_dir: str | Path, project: str, out_dir: str | Path,
               doc_type: str = _DEFAULT_TYPE) -> dict[str, Any]:
    """Export the compiled wiki of `project` as an OKF v0.1 bundle under `out_dir`.

    Non-destructive: reads only; the canonical wiki is untouched. Returns a summary
    {project, n_articles, out_dir, okf_version}."""
    vault_dir = Path(vault_dir)
    articles_dir = vault_dir / project / "wiki" / "articles"
    out = Path(out_dir)
    out_articles = out / "articles"
    out_articles.mkdir(parents=True, exist_ok=True)

    files = sorted(articles_dir.glob("*.md")) if articles_dir.exists() else []
    slugs = {f.stem for f in files}
    index_entries: list[tuple[str, str, str]] = []   # (slug, title, description)

    for f in files:
        md = f.read_text(encoding="utf-8", errors="replace")
        fm_text, body = _split_frontmatter(md)
        title = _extract_frontmatter_field(md, "title") or f.stem
        description = _extract_frontmatter_field(md, "summary_1line")
        fm = _okf_frontmatter(
            doc_type=doc_type, title=title, description=description,
            tags=_extract_frontmatter_tags(md),
            timestamp=_to_timestamp(_extract_frontmatter_field(md, "date")),
            resource=_extract_frontmatter_field(md, "resource"),   # usually absent in the wiki
            sources=_extract_sources(fm_text),
        )
        new_body = _rewrite_links(_strip_orphan_fence(body), slugs).lstrip("\n")
        (out_articles / f.name).write_text(fm + "\n\n" + new_body, encoding="utf-8")
        index_entries.append((f.stem, title, description))

    # OKF index.md — navigation with standard relative Markdown links (not [[wikilinks]])
    idx = ["---", "type: Index", f"title: {_yaml_dq(project)}", "---", "", f"# {project}", ""]
    for slug, title, desc in sorted(index_entries, key=lambda e: e[1].lower()):
        line = f"- [{title}](./articles/{slug}.md)"
        if desc:
            line += f" — {desc}"
        idx.append(line)
    (out / "index.md").write_text("\n".join(idx) + "\n", encoding="utf-8")

    return {"project": project, "n_articles": len(files),
            "out_dir": str(out), "okf_version": _OKF_VERSION}
