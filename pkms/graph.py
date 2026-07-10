"""Wiki link-graph domain logic (extracted from pkms.web — roadmap 1.12).

Builds the Obsidian-style graph for a project from the Markdown files on disk
(no persisted graph store) and caches it with mtime-based invalidation. Pure
domain code: no FastAPI, no request handling — testable without a web client.
See docs/pkms_component_diagram.md ("Link-graph cache") for the design notes.
"""

import re
import threading
from pathlib import Path
from typing import Any

from pkms.compiler import _extract_frontmatter_field, _extract_frontmatter_tags
from pkms.linter import _extract_wiki_links


def _norm_concept(text: str) -> str:
    """Canonical key for a concept so '[[energy-based-model]]', '[[energy-based model]]'
    and '[[Energy Based Model]]' collapse to one node (lowercase, separators → '-')."""
    return re.sub(r"[\s_-]+", "-", text.lower()).strip("-")


# (articles_dir, concepts?, tags?) → (stamp, graph). The stamp is the sorted
# (name, mtime_ns) of every article: a stat() sweep per request instead of
# reading + regex-parsing the whole wiki — which every article page view did,
# O(N articles), just to compute one slug's backlinks.
_graph_cache: dict[tuple[str, bool, bool], tuple[tuple, dict[str, Any]]] = {}
_graph_cache_lock = threading.Lock()


def _articles_stamp(articles_dir: Path) -> tuple:
    if not articles_dir.exists():
        return ()
    return tuple(sorted((p.name, p.stat().st_mtime_ns) for p in articles_dir.glob("*.md")))


def _wiki_link_graph(
    vault_dir: Path, project: str, include_concepts: bool = True, include_tags: bool = True
) -> dict[str, Any]:
    """Build an Obsidian-style graph for a project (cached on article mtimes).

    Node types: "article" (a compiled page), "concept" (an unresolved [[wikilink]]
    target — the placeholder concepts the LLM references), "tag" (a frontmatter
    tag). Edges: article→article (resolved links), article→concept, article→tag.
    Returns {"nodes":[{"id","label","type"}], "links":[{"source","target"}]}.
    Reuses linter._extract_wiki_links so web + linter share one regex.

    The cache invalidates when any article is added/removed/modified (mtime_ns
    stamp). Callers must treat the returned dict as read-only — it is shared.
    """
    articles_dir = vault_dir / project / "wiki" / "articles"
    cache_key = (str(articles_dir), include_concepts, include_tags)
    stamp = _articles_stamp(articles_dir)
    with _graph_cache_lock:
        hit = _graph_cache.get(cache_key)
        if hit is not None and hit[0] == stamp:
            return hit[1]

    graph = _build_wiki_link_graph(articles_dir, include_concepts, include_tags)
    with _graph_cache_lock:
        _graph_cache[cache_key] = (stamp, graph)
    return graph


def _build_wiki_link_graph(
    articles_dir: Path, include_concepts: bool, include_tags: bool
) -> dict[str, Any]:
    nodes: dict[str, dict[str, Any]] = {}   # id -> node
    links: list[dict[str, str]] = []

    def _add(node_id: str, label: str, node_type: str) -> None:
        nodes.setdefault(node_id, {"id": node_id, "label": label, "type": node_type})

    # First pass: which slugs are real articles (so links can resolve)
    files = sorted(articles_dir.glob("*.md")) if articles_dir.exists() else []
    slugs = {f.stem for f in files}

    for md_file in files:
        slug = md_file.stem
        try:
            md = md_file.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        _add(slug, _extract_frontmatter_field(md, "title") or slug, "article")

        for raw_target in _extract_wiki_links(md):
            t = raw_target.strip()
            t = Path(t).stem if "/" in t else t
            if not t:
                continue
            if t in slugs:
                if t != slug:
                    links.append({"source": slug, "target": t})        # article → article
            elif include_concepts:
                cid = "concept:" + _norm_concept(t)   # dedup variants; first label kept
                _add(cid, t, "concept")
                links.append({"source": slug, "target": cid})          # article → concept

        if include_tags:
            for tag in _extract_frontmatter_tags(md):
                tid = f"tag:{tag}"
                _add(tid, tag, "tag")
                links.append({"source": slug, "target": tid})          # article → tag

    return {"nodes": list(nodes.values()), "links": links}
