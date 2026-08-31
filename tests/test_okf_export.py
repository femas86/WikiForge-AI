"""OKF v0.1 export adapter — unit + end-to-end (read-only over the canonical wiki)."""
from pathlib import Path

import pkms.okf_export as okf


def test_to_timestamp_widens_date_and_passes_through():
    assert okf._to_timestamp("2025-03-10") == "2025-03-10T00:00:00Z"
    assert okf._to_timestamp("2025-03-10T14:30:00Z") == "2025-03-10T14:30:00Z"  # already full
    assert okf._to_timestamp("") == "" and okf._to_timestamp(None) == ""


def test_rewrite_links_resolved_anchor_and_unresolved():
    slugs = {"beta"}
    body = "See [[beta]], [[beta|Beta anchor]] and [[nowhere]]."
    out = okf._rewrite_links(body, slugs)
    assert "[beta](./beta.md)" in out            # resolved → relative md link
    assert "[Beta anchor](./beta.md)" in out     # anchor form keeps the anchor as label
    assert "nowhere" in out and "[[nowhere]]" not in out and "./nowhere.md" not in out  # unresolved → plain text


def test_rewrite_links_malformed_does_not_swallow_next_link():
    """A malformed source link (wrong closing delimiter) must NOT greedily consume the
    following well-formed link on the same line."""
    slugs = {"beta"}
    body = "init with [[Foo>>, then compared against [[beta]] and [[gamma]]."
    out = okf._rewrite_links(body, slugs)
    assert "[beta](./beta.md)" in out     # well-formed neighbour still resolves
    assert "gamma" in out                 # and the one after it is handled too
    assert out.count("[[") == 1           # only the malformed token remains, isolated


def test_strip_orphan_fence_only_when_unbalanced():
    assert okf._strip_orphan_fence("```\n\nReal text.\n") == "\nReal text.\n"   # orphan opener dropped
    kept = "```python\nx = 1\n```\nafter"
    assert okf._strip_orphan_fence(kept) == kept                              # balanced block kept
    bare_balanced = "```\ncode\n```\nafter"
    assert okf._strip_orphan_fence(bare_balanced) == bare_balanced           # balanced (even) kept
    assert okf._strip_orphan_fence("no fence here") == "no fence here"


def test_extract_sources_block_and_inline():
    block = "title: X\nsources:\n  - a.txt\n  - b.md\ndate: 2025-01-01"
    assert okf._extract_sources(block) == ["a.txt", "b.md"]
    inline = 'title: X\nsources: [a.txt, "b.md"]\n'
    assert okf._extract_sources(inline) == ["a.txt", "b.md"]
    assert okf._extract_sources("title: X\n") == []


def _make_wiki(tmp_path: Path) -> Path:
    arts = tmp_path / "vault" / "p" / "wiki" / "articles"
    arts.mkdir(parents=True)
    (arts / "alpha.md").write_text(
        "---\n"
        "title: Alpha Article\n"
        "tags:\n  - x\n  - y\n"
        "sources:\n  - src1.txt\n  - src2.txt\n"
        "date: 2025-03-10\n"
        'summary_1line: "the alpha summary"\n'
        "---\n\n"
        "Body links to [[beta]] and [[nowhere]] and [[beta|Beta anchor]].\n",
        encoding="utf-8")
    (arts / "beta.md").write_text(
        "---\ntitle: Beta\ntags: [z]\ndate: 2024-01-01\n---\n\nBeta body.\n",
        encoding="utf-8")
    return tmp_path


def test_export_okf_end_to_end(tmp_path):
    root = _make_wiki(tmp_path)
    canonical = (root / "vault" / "p" / "wiki" / "articles" / "alpha.md").read_text(encoding="utf-8")

    res = okf.export_okf(root / "vault", "p", root / "out")

    assert res["n_articles"] == 2 and res["okf_version"] == "0.1"
    alpha = (root / "out" / "articles" / "alpha.md").read_text(encoding="utf-8")
    # OKF frontmatter: type is first + required; mapped fields present
    assert alpha.startswith("---\ntype: Wiki Article\n")
    assert 'title: "Alpha Article"' in alpha
    assert 'description: "the alpha summary"' in alpha          # from summary_1line
    assert "timestamp: 2025-03-10T00:00:00Z" in alpha          # date widened
    assert "tags: [x, y]" in alpha
    assert "sources:\n  - src1.txt\n  - src2.txt" in alpha      # provenance kept as extension
    # body: [[..]] rewritten to relative links; unresolved concept downgraded to text
    assert "[beta](./beta.md)" in alpha and "[Beta anchor](./beta.md)" in alpha
    assert "[[" not in alpha and "./nowhere.md" not in alpha

    # beta has no summary/sources → those lines omitted, but type/title/timestamp present
    beta = (root / "out" / "articles" / "beta.md").read_text(encoding="utf-8")
    assert "type: Wiki Article" in beta and 'title: "Beta"' in beta
    assert "timestamp: 2024-01-01T00:00:00Z" in beta
    assert "description:" not in beta and "sources:" not in beta

    # OKF index.md with standard relative links (not [[wikilinks]])
    index = (root / "out" / "index.md").read_text(encoding="utf-8")
    assert "type: Index" in index
    assert "[Alpha Article](./articles/alpha.md)" in index
    assert "[Beta](./articles/beta.md)" in index

    # NON-DESTRUCTIVE: the canonical article is byte-for-byte unchanged
    assert (root / "vault" / "p" / "wiki" / "articles" / "alpha.md").read_text(encoding="utf-8") == canonical


def test_export_okf_handles_article_without_frontmatter(tmp_path):
    arts = tmp_path / "vault" / "p" / "wiki" / "articles"
    arts.mkdir(parents=True)
    (arts / "raw.md").write_text("Just a body, no frontmatter, links [[missing]].\n", encoding="utf-8")
    res = okf.export_okf(tmp_path / "vault", "p", tmp_path / "out")
    assert res["n_articles"] == 1
    out = (tmp_path / "out" / "articles" / "raw.md").read_text(encoding="utf-8")
    assert out.startswith("---\ntype: Wiki Article\n")   # type always emitted
    assert 'title: "raw"' in out                         # falls back to the slug
    assert "missing" in out and "[[" not in out


def test_export_okf_empty_project_writes_index(tmp_path):
    res = okf.export_okf(tmp_path / "vault", "nope", tmp_path / "out")
    assert res["n_articles"] == 0
    assert (tmp_path / "out" / "index.md").exists()      # still a valid (empty) bundle
