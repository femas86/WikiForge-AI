import json
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pytest

from pkms.compiler import (
    _discover_work,
    _extract_frontmatter_field,
    _extract_frontmatter_tags,
    _hash_text,
    _load_conventions,
    _project_of,
    _slug,
    _strip_code_fence,
    _wiki_path,
    compile,
)
from pkms.db import init_db, upsert_article_source, upsert_file


CONFIG = {
    "qdrant": {
        "host": "localhost",
        "port": 6333,
        "collections": {"raw": "raw", "wiki": "wiki", "outputs": "outputs"},
    },
    "embedding": {"dimension": 4},
    "chunking": {"max_tokens": 50},
    "ollama": {"host": "localhost", "port": 11434,
                "models": {"embedding": "nomic-embed-text"}},
    "llm_router": {
        "agents": {"compiler": "claude"},
        "fallback": {"claude": "ollama"},
        "models": {"claude": "claude-sonnet-4-6", "ollama": "mistral:7b"},
    },
    "compile": {"crosslink_threshold": 0.82, "post_lint": False},
}


# ── pure helpers ──────────────────────────────────────────────────────────────

def test_slug():
    assert _slug("vault/default/raw/attention_paper.pdf") == "attention_paper"


def test_wiki_path():
    assert _wiki_path("vault/default/raw/foo.pdf") == "vault/default/wiki/articles/foo.md"


def test_wiki_path_lands_in_correct_project():
    assert (
        _wiki_path("vault/og-mdai/raw/foo.pdf", "og-mdai")
        == "vault/og-mdai/wiki/articles/foo.md"
    )


def test_project_of_parses_vault_relative():
    assert _project_of("vault/og-mdai/raw/foo.pdf") == "og-mdai"


def test_project_of_raises_on_absolute_path():
    # Regression: an absolute path used to silently resolve to "default",
    # misplacing the compiled article. It must now raise loudly.
    with pytest.raises(ValueError):
        _project_of("/home/user/vault/og-mdai/raw/foo.pdf")


def test_wiki_path_raises_on_project_mismatch():
    # Path says og-mdai but compile was invoked for default → must not silently
    # write into the wrong project's wiki dir.
    with pytest.raises(ValueError):
        _wiki_path("vault/og-mdai/raw/foo.pdf", "default")


def test_hash_text_deterministic():
    assert _hash_text("hello") == _hash_text("hello")
    assert _hash_text("hello") != _hash_text("world")


def test_strip_code_fence_unwraps_markdown_frontmatter():
    # LLM wraps the whole article in a ```markdown fence, pushing the YAML
    # frontmatter off line 1 and breaking frontmatter parsing.
    wrapped = "```markdown\n---\ntitle: Foo\n---\n\nBody text.\n```"
    out = _strip_code_fence(wrapped)
    assert out.startswith("---\ntitle: Foo")
    assert not out.endswith("```")
    assert _extract_frontmatter_field(out, "title") == "Foo"


def test_strip_code_fence_leaves_plain_text_unchanged():
    plain = "---\ntitle: Foo\n---\n\nBody."
    assert _strip_code_fence(plain) == plain


def test_strip_code_fence_preserves_inner_fences():
    wrapped = "```markdown\n# H\n\n```python\nx=1\n```\n\ndone\n```"
    out = _strip_code_fence(wrapped)
    assert "```python" in out
    assert out.startswith("# H")


def test_rebuild_index_is_deterministic_no_llm(tmp_path):
    # _index.md is pure string formatting over slug + frontmatter summary —
    # it must never spend an LLM call (or expose a compile cycle to a 429).
    from pkms.compiler import _rebuild_index
    articles = tmp_path / "demo" / "wiki" / "articles"
    articles.mkdir(parents=True)
    (articles / "beta.md").write_text(
        "---\ntitle: Beta\nsummary_1line: Second topic\n---\nBody.", encoding="utf-8")
    (articles / "alpha.md").write_text(
        "---\ntitle: Alpha\n---\nBody.", encoding="utf-8")   # no summary → title

    with patch("pkms.compiler.complete") as mock_llm:
        _rebuild_index(tmp_path, "demo")

    mock_llm.assert_not_called()
    index = (tmp_path / "demo" / "wiki" / "_index.md").read_text(encoding="utf-8")
    assert index == "- [[alpha]] — Alpha\n- [[beta]] — Second topic\n"


def test_rebuild_index_noop_without_articles(tmp_path):
    from pkms.compiler import _rebuild_index
    _rebuild_index(tmp_path, "demo")   # no articles dir — must not raise
    assert not (tmp_path / "demo" / "wiki" / "_index.md").exists()


def test_extract_frontmatter_field():
    md = "---\ntitle: My Article\ntags: [a, b]\n---\n\nBody."
    assert _extract_frontmatter_field(md, "title") == "My Article"


def test_extract_frontmatter_tags():
    md = "---\ntags: [ml, nlp, transformers]\n---\n"
    assert _extract_frontmatter_tags(md) == ["ml", "nlp", "transformers"]


def test_extract_frontmatter_tags_empty():
    assert _extract_frontmatter_tags("no frontmatter") == []


def test_extract_frontmatter_tags_block_list():
    # The compiler emits YAML block-list tags; the parser must read them too.
    md = "---\ntitle: T\ntags:\n  - self-supervised-learning\n  - world-models\n  - meta-ai\nsources:\n  - x.pdf\n---\n"
    assert _extract_frontmatter_tags(md) == ["self-supervised-learning", "world-models", "meta-ai"]


# ── discover work ─────────────────────────────────────────────────────────────

def _seed_db(db_path):
    init_db(db_path)
    upsert_file(db_path, path="vault/default/raw/doc.pdf", hash="sha256:new",
                qdrant_ids=[], collection="raw", indexed_at="2026-06-11T10:00:00Z",
                project="default")


def test_discover_work_finds_uncompiled_source(tmp_path):
    db = str(tmp_path / "idx")
    _seed_db(db)
    work = _discover_work({"type": "full"}, db, CONFIG, "default")
    assert len(work) == 1
    assert work[0]["wiki_path"] == "vault/default/wiki/articles/doc.md"
    assert work[0]["sources"][0]["raw_path"] == "vault/default/raw/doc.pdf"


def test_discover_work_finds_stale_article(tmp_path):
    db = str(tmp_path / "idx")
    _seed_db(db)
    upsert_file(db, path="vault/default/wiki/articles/doc.md", hash="sha256:w",
                qdrant_ids=[], collection="wiki", indexed_at="2026-06-11T10:00:00Z",
                project="default")
    upsert_article_source(db, wiki_path="vault/default/wiki/articles/doc.md",
                          raw_path="vault/default/raw/doc.pdf",
                          compile_hash="sha256:OLD",  # stale
                          compiled_at="2026-06-11T10:00:00Z")
    work = _discover_work({"type": "full"}, db, CONFIG, "default")
    assert any(w["wiki_path"] == "vault/default/wiki/articles/doc.md" for w in work)


def test_discover_work_empty_when_current(tmp_path):
    db = str(tmp_path / "idx")
    _seed_db(db)
    upsert_file(db, path="vault/default/wiki/articles/doc.md", hash="sha256:w",
                qdrant_ids=[], collection="wiki", indexed_at="2026-06-11T10:00:00Z",
                project="default")
    upsert_article_source(db, wiki_path="vault/default/wiki/articles/doc.md",
                          raw_path="vault/default/raw/doc.pdf",
                          compile_hash="sha256:new",  # matches
                          compiled_at="2026-06-11T10:00:00Z")
    work = _discover_work({"type": "full"}, db, CONFIG, "default")
    assert work == []


def test_discover_work_doc_scope_filters(tmp_path):
    db = str(tmp_path / "idx")
    _seed_db(db)
    upsert_file(db, path="vault/default/raw/other.pdf", hash="sha256:o",
                qdrant_ids=[], collection="raw", indexed_at="2026-06-11T10:00:00Z",
                project="default")
    work = _discover_work({"type": "doc", "source": "vault/default/raw/doc.pdf"}, db, CONFIG, "default")
    assert len(work) == 1
    assert work[0]["sources"][0]["raw_path"] == "vault/default/raw/doc.pdf"


# ── compile happy path ────────────────────────────────────────────────────────

ARTICLE_MD = """\
---
title: "Doc Article"
tags: [ml, test]
sources: [vault/default/raw/doc.pdf]
date: 2026-06-11
summary_1line: A test article.
---

## Introduction

This is the article body.

## Sources

- vault/default/raw/doc.pdf
"""


def _setup_compile(tmp_path):
    """Set up vault structure and db for a compile test."""
    (tmp_path / "vault" / "default" / "raw").mkdir(parents=True)
    (tmp_path / "vault" / "default" / "wiki" / "articles").mkdir(parents=True)
    db = str(tmp_path / "idx")
    _seed_db(db)
    return db


def test_compile_skipped_when_nothing_to_do(tmp_path):
    db = str(tmp_path / "idx")
    init_db(db)  # empty db — no raw files
    result = compile({"type": "full"}, str(tmp_path), db, "token-123", CONFIG)
    assert result["status"] == "SKIPPED"


def test_compile_writes_article(tmp_path):
    db = _setup_compile(tmp_path)

    with patch("pkms.compiler.complete", return_value=ARTICLE_MD), \
         patch("pkms.embed._embed_batch", side_effect=lambda ts, c: [[0.1, 0.2, 0.3, 0.4]] * len(ts)), \
         patch("pkms.compiler.scroll", return_value=[
             {"id": "x", "payload": {"text": "chunk text", "path": "vault/default/raw/doc.pdf",
                                     "chunk_index": 0, "section_heading": ""}}
         ]), \
         patch("pkms.indexing.upsert_batch"), \
         patch("pkms.compiler.delete_by_ids"), \
         patch("pkms.compiler._git_commit"), \
         patch("pkms.compiler._rebuild_index"), \
         patch("pkms.compiler._crosslink_pass", return_value=0):
        result = compile({"type": "full"}, str(tmp_path), db, "token-123", CONFIG)

    assert result["status"] == "DONE"
    assert result["n_articles"] == 1
    article = tmp_path / "vault" / "default" / "wiki" / "articles" / "doc.md"
    assert article.exists()
    assert "Introduction" in article.read_text()


def test_compile_stores_chunk_text_in_wiki_payload(tmp_path):
    """Regression: the wiki Qdrant payload MUST carry chunk text so querier
    _format_hits can ground wiki-sourced answers in content, not just headings."""
    db = _setup_compile(tmp_path)

    with patch("pkms.compiler.complete", return_value=ARTICLE_MD), \
         patch("pkms.embed._embed_batch", side_effect=lambda ts, c: [[0.1, 0.2, 0.3, 0.4]] * len(ts)), \
         patch("pkms.compiler.scroll", return_value=[]), \
         patch("pkms.indexing.upsert_batch") as mock_upsert, \
         patch("pkms.compiler.delete_by_ids"), \
         patch("pkms.compiler._git_commit"), \
         patch("pkms.compiler._rebuild_index"), \
         patch("pkms.compiler._crosslink_pass", return_value=0):
        compile({"type": "full"}, str(tmp_path), db, "token-123", CONFIG)

    payloads = [pl for c in mock_upsert.call_args_list for (_pid, _vec, pl) in c.args[1]]
    assert payloads
    assert all(p.get("text") for p in payloads)     # every wiki chunk carries its text


def _seed_second_raw(db):
    from pkms.db import upsert_file
    upsert_file(db, path="vault/default/raw/doc2.pdf", hash="sha256:new2",
                qdrant_ids=[], collection="raw", indexed_at="2026-06-11T10:00:00Z", project="default")


def test_compile_skips_failing_doc_and_continues(tmp_path):
    """One doc that can't compile (e.g. rate limit + fallback both fail) must NOT
    abort the batch — it's skipped and reported, the others still compile."""
    db = _setup_compile(tmp_path)
    _seed_second_raw(db)

    def fake_compile(wiki_path, sources, vault_dir, config, conventions=""):
        if "doc2" in wiki_path:
            raise RuntimeError("Both primary 'claude' and fallback 'ollama' failed")
        return ARTICLE_MD

    with patch("pkms.compiler._compile_article", side_effect=fake_compile), \
         patch("pkms.embed._embed_batch", side_effect=lambda ts, c: [[0.1, 0.2, 0.3, 0.4]] * len(ts)), \
         patch("pkms.compiler.scroll", return_value=[]), \
         patch("pkms.indexing.upsert_batch"), \
         patch("pkms.compiler.delete_by_ids"), \
         patch("pkms.compiler._git_commit"), \
         patch("pkms.compiler._rebuild_index"), \
         patch("pkms.compiler._crosslink_pass", return_value=0):
        result = compile({"type": "full"}, str(tmp_path), db, "tok", CONFIG)

    assert result["status"] == "DONE"          # did not abort
    assert result["n_articles"] == 1           # the healthy doc compiled
    assert result["n_failed"] == 1
    assert any("doc2" in p for p in result["failed"])
    arts = tmp_path / "vault" / "default" / "wiki" / "articles"
    assert (arts / "doc.md").exists()
    assert not (arts / "doc2.md").exists()     # failed one left uncompiled


def test_compile_paces_between_docs(tmp_path):
    db = _setup_compile(tmp_path)
    _seed_second_raw(db)
    cfg = {**CONFIG, "compile": {**CONFIG["compile"], "pace_seconds": 5}}

    with patch("pkms.compiler.time.sleep") as mock_sleep, \
         patch("pkms.compiler.complete", return_value=ARTICLE_MD), \
         patch("pkms.embed._embed_batch", side_effect=lambda ts, c: [[0.1, 0.2, 0.3, 0.4]] * len(ts)), \
         patch("pkms.compiler.scroll", return_value=[]), \
         patch("pkms.indexing.upsert_batch"), \
         patch("pkms.compiler.delete_by_ids"), \
         patch("pkms.compiler._git_commit"), \
         patch("pkms.compiler._rebuild_index"), \
         patch("pkms.compiler._crosslink_pass", return_value=0):
        compile({"type": "full"}, str(tmp_path), db, "tok", cfg)

    mock_sleep.assert_called_once_with(5)      # 2 docs → exactly one inter-doc pause


# ── project schema (vault/{project}/CLAUDE.md) ────────────────────────────────

def test_load_conventions_reads_project_schema(tmp_path):
    vault_dir = tmp_path / "vault"
    (vault_dir / "default").mkdir(parents=True)
    (vault_dir / "default" / "CLAUDE.md").write_text("Write in British English.", encoding="utf-8")
    assert _load_conventions(vault_dir, "default") == "Write in British English."


def test_load_conventions_absent_returns_empty(tmp_path):
    vault_dir = tmp_path / "vault"
    (vault_dir / "default").mkdir(parents=True)
    assert _load_conventions(vault_dir, "default") == ""


def _compile_capturing_prompt(tmp_path, db):
    """Run a one-article compile and return the prompt passed to complete()."""
    mock_complete = MagicMock(return_value=ARTICLE_MD)
    with patch("pkms.compiler.complete", mock_complete), \
         patch("pkms.embed._embed_batch", side_effect=lambda ts, c: [[0.1, 0.2, 0.3, 0.4]] * len(ts)), \
         patch("pkms.compiler.scroll", return_value=[
             {"id": "x", "payload": {"text": "chunk text", "path": "vault/default/raw/doc.pdf",
                                     "chunk_index": 0, "section_heading": ""}}
         ]), \
         patch("pkms.indexing.upsert_batch"), \
         patch("pkms.compiler.delete_by_ids"), \
         patch("pkms.compiler._git_commit"), \
         patch("pkms.compiler._rebuild_index"), \
         patch("pkms.compiler._crosslink_pass", return_value=0):
        compile({"type": "full"}, str(tmp_path), db, "token-123", CONFIG)
    return mock_complete.call_args.args[1]  # complete(agent, prompt, config)


def test_compile_injects_project_conventions(tmp_path):
    db = _setup_compile(tmp_path)
    marker = "ZZTOP_CONVENTION_MARKER: always cite page numbers"
    (tmp_path / "vault" / "default" / "CLAUDE.md").write_text(marker, encoding="utf-8")

    prompt = _compile_capturing_prompt(tmp_path, db)
    assert marker in prompt
    assert "Project writing conventions" in prompt


def test_compile_without_conventions_omits_header(tmp_path):
    db = _setup_compile(tmp_path)
    prompt = _compile_capturing_prompt(tmp_path, db)
    assert "Project writing conventions" not in prompt


def test_compile_updates_search_index(tmp_path):
    db = _setup_compile(tmp_path)
    from pkms.db import get_file, get_article_sources

    with patch("pkms.compiler.complete", return_value=ARTICLE_MD), \
         patch("pkms.embed._embed_batch", side_effect=lambda ts, c: [[0.1, 0.2, 0.3, 0.4]] * len(ts)), \
         patch("pkms.compiler.scroll", return_value=[
             {"id": "x", "payload": {"text": "chunk", "path": "vault/default/raw/doc.pdf",
                                     "chunk_index": 0, "section_heading": ""}}
         ]), \
         patch("pkms.indexing.upsert_batch"), \
         patch("pkms.compiler.delete_by_ids"), \
         patch("pkms.compiler._git_commit"), \
         patch("pkms.compiler._rebuild_index"), \
         patch("pkms.compiler._crosslink_pass", return_value=0):
        compile({"type": "full"}, str(tmp_path), db, "token-123", CONFIG)

    wiki_row = get_file(db, "vault/default/wiki/articles/doc.md")
    assert wiki_row is not None
    assert wiki_row["collection"] == "wiki"

    sources = get_article_sources(db, "vault/default/wiki/articles/doc.md")
    assert len(sources) == 1
    assert sources[0]["compile_hash"] == "sha256:new"


def test_compile_doc_scope(tmp_path):
    db = _setup_compile(tmp_path)
    upsert_file(db, path="vault/default/raw/other.pdf", hash="sha256:o",
                qdrant_ids=[], collection="raw", indexed_at="2026-06-11T10:00:00Z",
                project="default")

    compiled_paths = []

    def fake_write_article(wiki_path, sources, vault_dir, cfg, conventions=""):
        compiled_paths.append(wiki_path)
        return ARTICLE_MD

    with patch("pkms.compiler._compile_article", side_effect=fake_write_article), \
         patch("pkms.compiler._embed_and_upsert_wiki", return_value=["id1"]), \
         patch("pkms.compiler._git_commit"), \
         patch("pkms.compiler._rebuild_index"), \
         patch("pkms.compiler._crosslink_pass", return_value=0):
        compile({"type": "doc", "source": "vault/default/raw/doc.pdf"},
                str(tmp_path), db, "token", CONFIG)

    assert compiled_paths == ["vault/default/wiki/articles/doc.md"]


def test_compile_guard_write_enforced(tmp_path):
    db = _setup_compile(tmp_path)

    with patch("pkms.compiler.complete", return_value=ARTICLE_MD), \
         patch("pkms.embed._embed_batch", side_effect=lambda ts, c: [[0.1, 0.2, 0.3, 0.4]] * len(ts)), \
         patch("pkms.compiler.scroll", return_value=[]), \
         patch("pkms.indexing.upsert_batch"), \
         patch("pkms.compiler.delete_by_ids"), \
         patch("pkms.compiler._git_commit"), \
         patch("pkms.compiler._rebuild_index"), \
         patch("pkms.compiler._crosslink_pass", return_value=0), \
         patch("pkms.compiler.guard_write") as mock_guard:
        compile({"type": "full"}, str(tmp_path), db, "token", CONFIG)

    # guard_write must have been called for the article write and db write
    assert mock_guard.call_count >= 2


# ── _git_commit auto-init ─────────────────────────────────────────────────────

def test_git_commit_initialises_repo_on_fresh_vault(tmp_path):
    import git as gitpython
    from pkms.compiler import _git_commit

    vault_dir = tmp_path / "vault"
    (vault_dir / "wiki" / "articles").mkdir(parents=True)
    article = vault_dir / "wiki" / "articles" / "topic.md"
    article.write_text("# Topic\n")

    _git_commit(vault_dir, ["wiki/articles/topic.md"], "compiler: add topic")

    repo = gitpython.Repo(str(vault_dir))  # raises if init didn't happen
    assert repo.head.is_valid()
    assert repo.head.commit.message.startswith("compiler: add topic")


def test_git_commit_commits_subsequent_changes(tmp_path):
    import git as gitpython
    from pkms.compiler import _git_commit

    vault_dir = tmp_path / "vault"
    (vault_dir / "wiki").mkdir(parents=True)
    f = vault_dir / "wiki" / "a.md"
    f.write_text("v1\n")
    _git_commit(vault_dir, ["wiki/a.md"], "first")
    f.write_text("v2\n")
    _git_commit(vault_dir, ["wiki/a.md"], "second")

    repo = gitpython.Repo(str(vault_dir))
    assert repo.head.commit.message.startswith("second")


# ── B2 hierarchical (map-reduce) compile ──────────────────────────────────────

def test_group_by_budget_packs_within_budget():
    from pkms.compiler import _group_by_budget
    units = ["x" * 40, "y" * 40, "z" * 40]   # ~10 est-tokens each (len//4)
    groups = _group_by_budget(units, 20)      # budget fits 2 units
    assert [len(g) for g in groups] == [2, 1]


def test_compile_article_single_pass_under_budget(tmp_path):
    from pkms.compiler import _compile_article
    vault_dir = tmp_path / "vault"
    (vault_dir / "default" / "wiki" / "articles").mkdir(parents=True)
    pts = [{"id": "0", "payload": {"text": "short body", "path": "vault/default/raw/doc.pdf",
                                   "chunk_index": 0, "section_heading": ""}}]
    calls = []
    def fake_complete(agent, prompt, config, **kw):
        calls.append(prompt)
        return ARTICLE_MD
    with patch("pkms.compiler.scroll", return_value=pts), \
         patch("pkms.compiler.complete", side_effect=fake_complete):
        _compile_article("vault/default/wiki/articles/doc.md",
                         [{"raw_path": "vault/default/raw/doc.pdf", "hash": "h"}], vault_dir, CONFIG)
    assert len(calls) == 1                                   # one call, no map-reduce
    assert "Extract the key factual content" not in calls[0]  # WRITE prompt, not MAP


def test_compile_article_mapreduce_when_over_budget(tmp_path):
    from pkms.compiler import _compile_article
    vault_dir = tmp_path / "vault"
    (vault_dir / "default" / "wiki" / "articles").mkdir(parents=True)
    cfg = {**CONFIG, "compile": {**CONFIG["compile"], "max_prompt_tokens": 20, "map_num_predict": 64}}
    # 4 chunks ~10 est-tokens each → total 40 > budget 20 → map-reduce (2 groups)
    pts = [{"id": str(i), "payload": {"text": "w" * 40, "path": "vault/default/raw/doc.pdf",
                                      "chunk_index": i, "section_heading": ""}} for i in range(4)]
    seen = {"map": 0, "reduce": 0}
    def fake_complete(agent, prompt, config, **kw):
        if "Extract the key factual content" in prompt:
            seen["map"] += 1
            return "DIGEST"
        if "extracted notes below" in prompt:
            seen["reduce"] += 1
            return "# FINAL ARTICLE"
        return "?"
    with patch("pkms.compiler.scroll", return_value=pts), \
         patch("pkms.compiler.complete", side_effect=fake_complete):
        out = _compile_article("vault/default/wiki/articles/doc.md",
                               [{"raw_path": "vault/default/raw/doc.pdf", "hash": "h"}], vault_dir, cfg)
    assert seen["map"] >= 1        # summarised group(s)
    assert seen["reduce"] == 1     # single reduce into the article
    assert out == "# FINAL ARTICLE"


# ── B: article output cap (num_predict) + truncation guard ────────────────────

def _compile_patches(article_md):
    return [
        patch("pkms.compiler.complete", return_value=article_md),
        patch("pkms.embed._embed_batch", side_effect=lambda ts, c: [[0.1, 0.2, 0.3, 0.4]] * len(ts)),
        patch("pkms.compiler.scroll", return_value=[
            {"id": "x", "payload": {"text": "chunk text", "path": "vault/default/raw/doc.pdf",
                                    "chunk_index": 0, "section_heading": ""}}]),
        patch("pkms.indexing.upsert_batch"),
        patch("pkms.compiler.delete_by_ids"),
        patch("pkms.compiler._git_commit"),
        patch("pkms.compiler._rebuild_index"),
        patch("pkms.compiler._crosslink_pass", return_value=0),
    ]


def test_compile_passes_article_num_predict(tmp_path):
    db = _setup_compile(tmp_path)
    cfg = {**CONFIG, "compile": {**CONFIG["compile"], "article_num_predict": 7777}}
    patches = _compile_patches(ARTICLE_MD)
    with patches[0] as mock_complete, patches[1], patches[2], patches[3], \
         patches[4], patches[5], patches[6], patches[7]:
        compile({"type": "full"}, str(tmp_path), db, "tok", cfg)
    # the article-generation call carried the configured output cap (7777 is unique vs crosslink's default)
    assert 7777 in [c.kwargs.get("num_predict") for c in mock_complete.call_args_list]


def test_compile_warns_on_truncated_article(tmp_path, caplog):
    db = _setup_compile(tmp_path)
    truncated = "---\ntitle: X\ntags: [a]\nsources: [s]\ndate: 2026\n---\n\n## Intro\n\nBody cut off mid-sen"
    patches = _compile_patches(truncated)   # no "## Sources" → truncation guard fires
    with patches[0], patches[1], patches[2], patches[3], patches[4], patches[5], patches[6], patches[7]:
        with caplog.at_level("WARNING", logger="pkms.compiler"):
            compile({"type": "full"}, str(tmp_path), db, "tok", CONFIG)
    assert any("looks truncated" in r.message for r in caplog.records)


# ── B9 output-side chunking: section-by-section compile ───────────────────────

def test_compile_sectioned_assembles_complete_article():
    from pkms.compiler import _compile_sectioned, _looks_truncated
    outline = json.dumps({
        "frontmatter": {"title": "BERT", "summary_1line": "A masked language model.", "tags": ["nlp", "transformers"]},
        "sections": ["Introduction", "Architecture"],
    })
    seq = [outline, "## Introduction\n\nBERT pre-trains deep bidirectional representations.",
           "## Architecture\n\nA multi-layer Transformer encoder."]
    with patch("pkms.compiler.complete", side_effect=seq):
        art = _compile_sectioned("faithful notes about BERT",
                                 [{"raw_path": "vault/default/raw/bert.pdf"}], "", CONFIG, "bert")
    assert art.startswith("---")                       # frontmatter
    assert 'title: "BERT"' in art
    assert "## Introduction" in art and "## Architecture" in art   # every section present
    assert "## Sources" in art and "vault/default/raw/bert.pdf" in art
    assert not _looks_truncated(art)                   # complete by construction


def test_compile_sectioned_returns_none_on_bad_outline():
    from pkms.compiler import _compile_sectioned
    with patch("pkms.compiler.complete", return_value="not valid json"):
        out = _compile_sectioned("notes", [{"raw_path": "x.pdf"}], "", CONFIG, "t")
    assert out is None                                 # caller falls back to single-pass


def test_compile_sectioned_dedupes_sources_deterministically():
    from pkms.compiler import _compile_sectioned
    outline = json.dumps({"frontmatter": {"title": "T", "summary_1line": "s", "tags": []},
                          "sections": ["Only"]})
    with patch("pkms.compiler.complete", side_effect=[outline, "## Only\n\nbody"]):
        art = _compile_sectioned("notes", [{"raw_path": "a.pdf"}, {"raw_path": "a.pdf"},
                                           {"raw_path": "b.pdf"}], "", CONFIG, "t")
    # Sources section built from the sources arg (deterministic), not the LLM
    assert art.count("- a.pdf") == 1 and "- b.pdf" in art
