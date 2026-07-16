from pathlib import Path
from unittest.mock import patch

import pytest

from pkms.db import init_db, upsert_article_source, upsert_file
from pkms.linter import (
    _check_broken_links,
    _check_drift,
    _check_frontmatter,
    _check_missing_files,
    _check_orphans,
    _extract_wiki_links,
    _parse_frontmatter_keys,
    lint,
)

CONFIG = {
    "qdrant": {"host": "localhost", "port": 6333,
               "collections": {"raw": "raw", "wiki": "wiki", "outputs": "outputs"}},
    "embedding": {"dimension": 4},
    "chunking": {"max_tokens": 200},
}


# ── _extract_wiki_links ───────────────────────────────────────────────────────

def test_extract_simple_link():
    assert _extract_wiki_links("See [[transformers]] for details.") == ["transformers"]


def test_extract_link_with_anchor():
    assert _extract_wiki_links("Read [[transformers|the article]].") == ["transformers"]


def test_extract_multiple_links():
    result = _extract_wiki_links("[[a]] and [[b|B text]] and [[c]]")
    assert result == ["a", "b", "c"]


def test_extract_no_links():
    assert _extract_wiki_links("No links here.") == []


# ── _parse_frontmatter_keys ───────────────────────────────────────────────────

def test_parse_frontmatter_keys_full():
    md = "---\ntitle: T\ntags: [a]\nsources: [x]\ndate: 2026-06-11\n---\n"
    keys = _parse_frontmatter_keys(md)
    assert keys == {"title", "tags", "sources", "date"}


def test_parse_frontmatter_keys_missing():
    md = "---\ntitle: T\n---\n"
    keys = _parse_frontmatter_keys(md)
    assert "tags" not in keys


def test_parse_frontmatter_no_block():
    assert _parse_frontmatter_keys("no frontmatter") == set()


# ── _check_drift ─────────────────────────────────────────────────────────────

def _seed_stale(tmp_path):
    db = str(tmp_path / "idx")
    init_db(db)
    upsert_file(db, path="vault/default/raw/doc.pdf", hash="sha256:NEW",
                qdrant_ids=[], collection="raw", indexed_at="2026-06-11T10:00:00Z",
                project="default")
    upsert_file(db, path="vault/default/wiki/articles/doc.md", hash="sha256:w",
                qdrant_ids=[], collection="wiki", indexed_at="2026-06-11T10:00:00Z",
                project="default")
    upsert_article_source(db, wiki_path="vault/default/wiki/articles/doc.md",
                          raw_path="vault/default/raw/doc.pdf",
                          compile_hash="sha256:OLD",
                          compiled_at="2026-06-11T09:00:00Z")
    return db


def test_check_drift_detects_stale(tmp_path):
    db = _seed_stale(tmp_path)
    issues = _check_drift(db, "default")
    assert len(issues) == 1
    assert issues[0]["type"] == "DRIFT"
    assert issues[0]["wiki_path"] == "vault/default/wiki/articles/doc.md"


def test_check_drift_clean(tmp_path):
    db = str(tmp_path / "idx")
    init_db(db)
    upsert_file(db, path="vault/default/raw/doc.pdf", hash="sha256:X",
                qdrant_ids=[], collection="raw", indexed_at="2026-06-11T10:00:00Z",
                project="default")
    upsert_file(db, path="vault/default/wiki/articles/doc.md", hash="sha256:w",
                qdrant_ids=[], collection="wiki", indexed_at="2026-06-11T10:00:00Z",
                project="default")
    upsert_article_source(db, wiki_path="vault/default/wiki/articles/doc.md",
                          raw_path="vault/default/raw/doc.pdf",
                          compile_hash="sha256:X",
                          compiled_at="2026-06-11T10:00:00Z")
    assert _check_drift(db, "default") == []


# ── _check_broken_links ───────────────────────────────────────────────────────

def _make_vault(tmp_path):
    vault_dir = tmp_path / "vault"
    (vault_dir / "default" / "wiki" / "articles").mkdir(parents=True)
    (vault_dir / "default" / "outputs").mkdir(parents=True)
    return vault_dir


def test_check_broken_links_detects_missing_target(tmp_path):
    vault_dir = _make_vault(tmp_path)
    (vault_dir / "default" / "wiki" / "articles" / "a.md").write_text(
        "See [[nonexistent]] for details."
    )
    issues = _check_broken_links(vault_dir, "default")
    assert len(issues) == 1
    assert issues[0]["type"] == "BROKEN_LINK"
    assert issues[0]["target"] == "nonexistent"


def test_check_broken_links_valid_link(tmp_path):
    vault_dir = _make_vault(tmp_path)
    (vault_dir / "default" / "wiki" / "articles" / "a.md").write_text("See [[b]] for details.")
    (vault_dir / "default" / "wiki" / "articles" / "b.md").write_text("# B article")
    assert _check_broken_links(vault_dir, "default") == []


def test_check_broken_links_no_articles_dir(tmp_path):
    vault_dir = tmp_path / "vault"
    vault_dir.mkdir()
    assert _check_broken_links(vault_dir, "default") == []


# ── _check_orphans ────────────────────────────────────────────────────────────

def test_check_orphans_detects_missing_source(tmp_path):
    vault_dir = _make_vault(tmp_path)
    db = str(tmp_path / "idx")
    init_db(db)
    upsert_file(db, path="vault/default/raw/gone.pdf", hash="sha256:x",
                qdrant_ids=[], collection="raw", indexed_at="2026-06-11T10:00:00Z",
                project="default")
    upsert_file(db, path="vault/default/wiki/articles/gone.md", hash="sha256:w",
                qdrant_ids=[], collection="wiki", indexed_at="2026-06-11T10:00:00Z",
                project="default")
    upsert_article_source(db, wiki_path="vault/default/wiki/articles/gone.md",
                          raw_path="vault/default/raw/gone.pdf",
                          compile_hash="sha256:x",
                          compiled_at="2026-06-11T10:00:00Z")
    # Do NOT create vault/default/raw/gone.pdf on disk → orphan
    issues = _check_orphans(db, vault_dir, "default")
    assert len(issues) == 1
    assert issues[0]["type"] == "ORPHAN"


def test_check_orphans_source_exists(tmp_path):
    vault_dir = _make_vault(tmp_path)
    (tmp_path / "vault" / "default" / "raw").mkdir(parents=True)
    (tmp_path / "vault" / "default" / "raw" / "doc.pdf").write_bytes(b"content")
    db = str(tmp_path / "idx")
    init_db(db)
    upsert_file(db, path="vault/default/raw/doc.pdf", hash="sha256:x",
                qdrant_ids=[], collection="raw", indexed_at="2026-06-11T10:00:00Z",
                project="default")
    upsert_file(db, path="vault/default/wiki/articles/doc.md", hash="sha256:w",
                qdrant_ids=[], collection="wiki", indexed_at="2026-06-11T10:00:00Z",
                project="default")
    upsert_article_source(db, wiki_path="vault/default/wiki/articles/doc.md",
                          raw_path="vault/default/raw/doc.pdf",
                          compile_hash="sha256:x",
                          compiled_at="2026-06-11T10:00:00Z")
    assert _check_orphans(db, vault_dir, "default") == []


# ── _check_missing_files ──────────────────────────────────────────────────────

def test_check_missing_files_detects_deleted(tmp_path):
    vault_dir = _make_vault(tmp_path)
    db = str(tmp_path / "idx")
    init_db(db)
    upsert_file(db, path="vault/default/raw/deleted.pdf", hash="sha256:x",
                qdrant_ids=[], collection="raw", indexed_at="2026-06-11T10:00:00Z",
                project="default")
    issues = _check_missing_files(db, vault_dir, "default")
    assert len(issues) == 1
    assert issues[0]["type"] == "MISSING_FILE"
    assert issues[0]["path"] == "vault/default/raw/deleted.pdf"


def test_check_missing_files_all_present(tmp_path):
    vault_dir = _make_vault(tmp_path)
    (tmp_path / "vault" / "default" / "raw").mkdir(parents=True)
    (tmp_path / "vault" / "default" / "raw" / "doc.pdf").write_bytes(b"x")
    db = str(tmp_path / "idx")
    init_db(db)
    upsert_file(db, path="vault/default/raw/doc.pdf", hash="sha256:x",
                qdrant_ids=[], collection="raw", indexed_at="2026-06-11T10:00:00Z",
                project="default")
    assert _check_missing_files(db, vault_dir, "default") == []


# ── _check_frontmatter ────────────────────────────────────────────────────────

def test_check_frontmatter_detects_missing_fields(tmp_path):
    vault_dir = _make_vault(tmp_path)
    (vault_dir / "default" / "wiki" / "articles" / "a.md").write_text("---\ntitle: T\n---\n\nBody.")
    issues = _check_frontmatter(vault_dir, "default")
    assert len(issues) == 1
    assert issues[0]["type"] == "BAD_FRONTMATTER"
    assert "tags" in issues[0]["missing_fields"]


def test_check_frontmatter_all_fields_present(tmp_path):
    vault_dir = _make_vault(tmp_path)
    (vault_dir / "default" / "wiki" / "articles" / "a.md").write_text(
        "---\ntitle: T\ntags: [x]\nsources: [y]\ndate: 2026-06-11\n---\n\nBody."
    )
    assert _check_frontmatter(vault_dir, "default") == []


# ── lint integration ──────────────────────────────────────────────────────────

def test_lint_clean_vault(tmp_path):
    vault_dir = _make_vault(tmp_path)
    db = str(tmp_path / "idx")
    init_db(db)
    result = lint(str(tmp_path), db, CONFIG)
    assert result["total_issues"] == 0
    assert "No issues found" in result["report_md"]


def test_lint_writes_report_file(tmp_path):
    vault_dir = _make_vault(tmp_path)
    db = str(tmp_path / "idx")
    init_db(db)
    result = lint(str(tmp_path), db, CONFIG)
    report_file = tmp_path / "vault" / "default" / "outputs" / "lint_report.md"
    assert report_file.exists()
    assert "Lint Report" in report_file.read_text()


def test_lint_reports_drift(tmp_path):
    _make_vault(tmp_path)
    db = _seed_stale(tmp_path)
    result = lint(str(tmp_path), db, CONFIG)
    assert result["issues_by_type"]["drift"]
    assert result["total_issues"] >= 1
    assert "Drift" in result["report_md"]


def test_lint_reports_broken_link(tmp_path):
    vault_dir = _make_vault(tmp_path)
    (vault_dir / "default" / "wiki" / "articles" / "a.md").write_text(
        "---\ntitle: A\ntags: []\nsources: []\ndate: 2026-06-11\n---\n\nSee [[missing]]."
    )
    db = str(tmp_path / "idx")
    init_db(db)
    result = lint(str(tmp_path), db, CONFIG)
    assert any(i["type"] == "BROKEN_LINK" for i in result["issues_by_type"]["broken_links"])


def test_lint_guard_write_called(tmp_path):
    _make_vault(tmp_path)
    db = str(tmp_path / "idx")
    init_db(db)
    with patch("pkms.linter.guard_write") as mock_guard:
        lint(str(tmp_path), db, CONFIG)
    mock_guard.assert_called_once()
    assert mock_guard.call_args[0][0] == "linter"


# ── B1: LLM semantic audit ────────────────────────────────────────────────────

import json as _json
from unittest.mock import MagicMock


def _two_articles(tmp_path):
    vault_dir = _make_vault(tmp_path)
    arts = vault_dir / "default" / "wiki" / "articles"
    (arts / "alpha.md").write_text(
        "---\ntitle: Alpha\nsummary_1line: About X\n---\n\nX was released in 2020.", encoding="utf-8")
    (arts / "beta.md").write_text(
        "---\ntitle: Beta\nsummary_1line: Also about X\n---\n\nX was released in 2022.", encoding="utf-8")
    return vault_dir


def test_semantic_audit_off_by_default_no_llm_call(tmp_path):
    _two_articles(tmp_path)
    db = str(tmp_path / "idx.db"); init_db(db)
    with patch("pkms.linter.complete") as mock_llm:
        result = lint(str(tmp_path), db, CONFIG, project="default")   # semantic=None, no config flag
    mock_llm.assert_not_called()
    assert result["issues_by_type"]["semantic"] == []


def test_semantic_audit_parses_findings_when_enabled(tmp_path):
    vault_dir = _two_articles(tmp_path)
    db = str(tmp_path / "idx.db"); init_db(db)
    finding = {"findings": [{"kind": "contradiction", "severity": "ERROR",
                             "articles": ["alpha", "beta"],
                             "detail": "alpha says 2020, beta says 2022"}]}
    with patch("pkms.linter.complete", return_value=_json.dumps(finding)) as mock_llm:
        result = lint(str(tmp_path), db, CONFIG, project="default", semantic=True)
    mock_llm.assert_called()                       # LLM audit ran
    sem = result["issues_by_type"]["semantic"]
    assert len(sem) == 1 and sem[0]["kind"] == "contradiction"
    assert sem[0]["articles"] == ["alpha", "beta"]
    assert "Semantic audit (LLM)" in result["report_md"]
    assert "contradiction" in result["report_md"]


def test_semantic_audit_config_flag_enables_it(tmp_path):
    _two_articles(tmp_path)
    db = str(tmp_path / "idx.db"); init_db(db)
    cfg = {**CONFIG, "lint": {"llm_audit": True}}
    with patch("pkms.linter.complete", return_value='{"findings": []}') as mock_llm:
        lint(str(tmp_path), db, cfg, project="default")   # semantic=None → reads config
    mock_llm.assert_called()


def test_semantic_audit_batches_under_budget(tmp_path):
    # two large articles + a budget whose usable room (budget-800, floored at 500)
    # fits one but not both → one LLM call per article
    vault_dir = _make_vault(tmp_path)
    arts = vault_dir / "default" / "wiki" / "articles"
    big = "word " * 400   # ~2000 chars → capped to 1500 → ~385 tokens per digest unit
    (arts / "alpha.md").write_text(f"---\ntitle: Alpha\n---\n\n{big}", encoding="utf-8")
    (arts / "beta.md").write_text(f"---\ntitle: Beta\n---\n\n{big}", encoding="utf-8")
    db = str(tmp_path / "idx.db"); init_db(db)
    cfg = {**CONFIG, "lint": {"semantic_max_prompt_tokens": 1500}}  # usable 700; one ~385-tok unit fits, two don't
    with patch("pkms.linter.complete", return_value='{"findings": []}') as mock_llm:
        lint(str(tmp_path), db, cfg, project="default", semantic=True)
    assert mock_llm.call_count == 2                 # one batch per article


def test_semantic_audit_survives_bad_json(tmp_path):
    _two_articles(tmp_path)
    db = str(tmp_path / "idx.db"); init_db(db)
    with patch("pkms.linter.complete", return_value="not json at all"):
        result = lint(str(tmp_path), db, CONFIG, project="default", semantic=True)
    assert result["issues_by_type"]["semantic"] == []   # unparseable → no findings, no crash


def test_semantic_digest_marks_clip_only_for_long_bodies(tmp_path):
    """Bug A regression: a body longer than body_chars is clipped WITH the marker
    (so the auditor won't mistake OUR cut for a defect); a short body is untouched."""
    from pkms.linter import _article_digest_units, _CLIP_MARKER
    vault_dir = _make_vault(tmp_path)
    arts = vault_dir / "default" / "wiki" / "articles"
    (arts / "short.md").write_text("---\ntitle: Short\n---\n\nTiny complete body.\n\n## Sources\n- a.md",
                                   encoding="utf-8")
    (arts / "long.md").write_text("---\ntitle: Long\n---\n\n" + ("lots of real content. " * 100),
                                  encoding="utf-8")
    units = _article_digest_units(vault_dir, "default", body_chars=200)
    joined = "\n".join(units)
    long_unit = next(u for u in units if u.startswith("### long"))
    short_unit = next(u for u in units if u.startswith("### short"))
    assert _CLIP_MARKER in long_unit        # long body clipped + marked
    assert _CLIP_MARKER not in short_unit    # short body shown whole, no marker
