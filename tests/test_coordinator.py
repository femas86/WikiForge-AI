import sys
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pytest

from pkms.coordinator import (
    handle_compile,
    handle_ingest,
    handle_lint,
    handle_query,
    _write_ingest_result,
    _db_path,
)
from pkms.db import init_db, get_file
from pkms.lock import LockTimeout

# ── minimal config ────────────────────────────────────────────────────────────

CFG = {
    "watcher": {"enabled": True, "debounce_seconds": 0.05, "watch_paths": ["raw/"]},
    "ingest": {"supported_extensions": [".md", ".pdf", ".html", ".txt"], "auto_compile": False},
    "qdrant": {"host": "localhost", "port": 6333,
               "collections": {"raw": "raw", "wiki": "wiki", "outputs": "outputs"}},
    "embedding": {"dimension": 4},
    "chunking": {"max_tokens": 200},
    "compile": {"crosslink_threshold": 0.82, "post_lint": False},
    "query": {"top_k_wiki": 3, "top_k_raw": 3, "mem0_recall": 2},
    "lock": {"ttl_seconds": 300, "max_wait_seconds": 1},
}

INGEST_RESULT = {
    "status": "DONE",
    "path": "vault/raw/doc.pdf",
    "hash": "sha256:abc",
    "qdrant_ids": ["id1"],
    "collection": "raw",
    "n_chunks": 3,
    "indexed_at": "2026-06-11T10:00:00Z",
}

SKIPPED_RESULT = {**INGEST_RESULT, "status": "SKIPPED"}


# ── _write_ingest_result ──────────────────────────────────────────────────────

def test_write_ingest_result_stores_file(tmp_path):
    vault = tmp_path
    (vault / "vault").mkdir()
    db = str(vault / "vault" / ".search-index")
    init_db(db)

    with patch("pkms.coordinator.guard_write"), patch("pkms.coordinator._git_commit"):
        _write_ingest_result(INGEST_RESULT, str(vault), db)

    row = get_file(db, "vault/raw/doc.pdf")
    assert row is not None
    assert row["hash"] == "sha256:abc"


def test_write_ingest_result_calls_guard_write(tmp_path):
    vault = tmp_path
    (vault / "vault").mkdir()
    db = str(vault / "vault" / ".search-index")
    init_db(db)

    with patch("pkms.coordinator.guard_write") as mock_guard, \
         patch("pkms.coordinator._git_commit"):
        _write_ingest_result(INGEST_RESULT, str(vault), db)

    mock_guard.assert_called_once()
    assert mock_guard.call_args[0][0] == "coordinator"


def test_write_ingest_result_commits_raw_to_vault_git(tmp_path):
    """Raw documents are committed AT INGEST (audit trail, §5.1) — not only the compiled
    articles, and not merely swept in later by an un-ingest's `git add -A`."""
    vault = tmp_path
    (vault / "vault").mkdir()
    db = str(vault / "vault" / ".search-index")
    init_db(db)

    with patch("pkms.coordinator.guard_write"), \
         patch("pkms.coordinator._git_commit") as mock_commit:
        _write_ingest_result({**INGEST_RESULT, "project": "og-mdai"}, str(vault), db)

    mock_commit.assert_called_once()
    vault_dir, rel_paths, message = mock_commit.call_args[0]
    assert vault_dir == vault / "vault"
    assert rel_paths == ["raw/doc.pdf"]            # vault-relative, like the compile commit
    assert message == "ingest: doc.pdf (og-mdai)"


# ── handle_ingest ─────────────────────────────────────────────────────────────

def test_handle_ingest_file_calls_ingest(tmp_path):
    vault = tmp_path
    (vault / "vault").mkdir()
    with patch("pkms.coordinator.ingest", return_value=INGEST_RESULT) as mock_ingest, \
         patch("pkms.coordinator._write_ingest_result") as mock_write, \
         patch("pkms.coordinator.init_db"):
        result = handle_ingest("/path/to/doc.pdf", str(vault), CFG)

    mock_ingest.assert_called_once()
    mock_write.assert_called_once()
    assert result["status"] == "DONE"


def test_handle_ingest_url_calls_fetch_and_ingest(tmp_path):
    vault = tmp_path
    (vault / "vault").mkdir()
    with patch("pkms.coordinator.fetch_and_ingest", return_value=INGEST_RESULT) as mock_fetch, \
         patch("pkms.coordinator._write_ingest_result"), \
         patch("pkms.coordinator.init_db"):
        handle_ingest("https://arxiv.org/abs/1234.5678", str(vault), CFG)

    mock_fetch.assert_called_once()


def test_handle_ingest_skipped_does_not_write(tmp_path):
    vault = tmp_path
    (vault / "vault").mkdir()
    with patch("pkms.coordinator.ingest", return_value=SKIPPED_RESULT), \
         patch("pkms.coordinator._write_ingest_result") as mock_write, \
         patch("pkms.coordinator.init_db"):
        handle_ingest("/path/to/doc.pdf", str(vault), CFG)

    mock_write.assert_not_called()


def test_handle_ingest_auto_compile_triggered(tmp_path):
    vault = tmp_path
    (vault / "vault").mkdir()
    cfg = {**CFG, "ingest": {**CFG["ingest"], "auto_compile": True}}
    with patch("pkms.coordinator.ingest", return_value=INGEST_RESULT), \
         patch("pkms.coordinator._write_ingest_result"), \
         patch("pkms.coordinator.init_db"), \
         patch("pkms.coordinator.handle_compile") as mock_compile:
        handle_ingest("/path/to/doc.pdf", str(vault), cfg)

    mock_compile.assert_called_once_with(scope="all", vault_root=str(vault), config=cfg, project="default")


def test_handle_ingest_auto_compile_not_triggered_on_skip(tmp_path):
    vault = tmp_path
    (vault / "vault").mkdir()
    cfg = {**CFG, "ingest": {**CFG["ingest"], "auto_compile": True}}
    with patch("pkms.coordinator.ingest", return_value=SKIPPED_RESULT), \
         patch("pkms.coordinator._write_ingest_result"), \
         patch("pkms.coordinator.init_db"), \
         patch("pkms.coordinator.handle_compile") as mock_compile:
        handle_ingest("/path/to/doc.pdf", str(vault), cfg)

    mock_compile.assert_not_called()


# ── handle_compile ────────────────────────────────────────────────────────────

def test_handle_compile_acquires_and_releases_wiki_lock(tmp_path):
    vault = tmp_path
    (vault / "vault").mkdir()
    compile_result = {"status": "DONE", "articles_written": 2}

    with patch("pkms.lock.init_locks_db"), \
         patch("pkms.lock.acquire", return_value="tok-123") as mock_acq, \
         patch("pkms.lock.release") as mock_rel, \
         patch("pkms.coordinator.run_compile", return_value=compile_result), \
         patch("pkms.coordinator.init_db"):
        handle_compile(scope="all", vault_root=str(vault), config=CFG)

    mock_acq.assert_called_once_with("wiki:default", "compiler", mock_acq.call_args[0][2], CFG)
    mock_rel.assert_called_once()
    assert mock_rel.call_args[0][1] == "tok-123"


def test_handle_compile_releases_lock_on_exception(tmp_path):
    vault = tmp_path
    (vault / "vault").mkdir()

    with patch("pkms.lock.init_locks_db"), \
         patch("pkms.lock.acquire", return_value="tok-err"), \
         patch("pkms.lock.release") as mock_rel, \
         patch("pkms.coordinator.run_compile", side_effect=RuntimeError("oops")), \
         patch("pkms.coordinator.init_db"):
        with pytest.raises(RuntimeError):
            handle_compile(scope="all", vault_root=str(vault), config=CFG)

    mock_rel.assert_called_once()


def test_handle_compile_raises_lock_timeout(tmp_path):
    vault = tmp_path
    (vault / "vault").mkdir()

    with patch("pkms.lock.init_locks_db"), \
         patch("pkms.lock.acquire", side_effect=LockTimeout("wiki")):
        with pytest.raises(LockTimeout):
            handle_compile(scope="all", vault_root=str(vault), config=CFG)


# ── handle_reindex ────────────────────────────────────────────────────────────

def test_handle_reindex_force_reingests_and_recompiles(tmp_path):
    from pkms.coordinator import handle_reindex
    from pkms.db import init_db, upsert_file, upsert_article_source, get_article_sources
    vault = tmp_path
    (vault / "vault").mkdir()
    db = str(vault / "vault" / ".search-index")
    init_db(db)
    # two raw docs, each with a compiled article
    for name in ("a", "b"):
        upsert_file(db, path=f"vault/p/raw/{name}.md", hash=f"h{name}", qdrant_ids=[f"r{name}"],
                    collection="raw", indexed_at="t", project="p")
        upsert_file(db, path=f"vault/p/wiki/articles/{name}.md", hash="hw", qdrant_ids=[f"w{name}"],
                    collection="wiki", indexed_at="t", project="p")
        upsert_article_source(db, wiki_path=f"vault/p/wiki/articles/{name}.md",
                              raw_path=f"vault/p/raw/{name}.md", compile_hash=f"h{name}", compiled_at="t")

    done = {"status": "DONE", "path": "x", "hash": "h", "content_hash": "c",
            "qdrant_ids": [], "collection": "raw", "indexed_at": "t", "project": "p"}
    with patch("pkms.coordinator.ingest", return_value=done) as mock_ingest, \
         patch("pkms.coordinator._write_ingest_result"), \
         patch("pkms.lock.init_locks_db"), \
         patch("pkms.lock.acquire", return_value="tok"), \
         patch("pkms.lock.release"), \
         patch("pkms.coordinator._delete_wiki_article") as mock_del_art, \
         patch("pkms.coordinator.handle_compile",
               return_value={"status": "DONE", "n_articles": 2, "failed": []}) as mock_compile:
        result = handle_reindex("p", str(vault), CFG)

    # every raw doc force re-ingested
    assert mock_ingest.call_count == 2
    assert all(c.kwargs.get("force") is True for c in mock_ingest.call_args_list)
    # both articles torn down, then one full recompile
    assert mock_del_art.call_count == 2
    mock_compile.assert_called_once_with(scope="all", vault_root=str(vault), config=CFG, project="p")
    assert result["reingested"] == 2 and result["articles_rebuilt"] == 2
    assert result["failed"] == []
    # article_sources actually cleared so the recompile treats raw as uncompiled
    assert get_article_sources(db, "vault/p/wiki/articles/a.md") == []


def test_handle_reindex_skips_empty_project(tmp_path):
    from pkms.coordinator import handle_reindex
    from pkms.db import init_db
    vault = tmp_path
    (vault / "vault").mkdir()
    init_db(str(vault / "vault" / ".search-index"))
    with patch("pkms.coordinator.ingest") as mock_ingest, \
         patch("pkms.coordinator.handle_compile") as mock_compile:
        result = handle_reindex("emptyproj", str(vault), CFG)
    assert result["status"] == "SKIPPED"
    mock_ingest.assert_not_called()
    mock_compile.assert_not_called()


# ── handle_query ──────────────────────────────────────────────────────────────

def test_handle_query_calls_run_query(tmp_path):
    vault = tmp_path
    (vault / "vault").mkdir()
    query_result = {
        "answer_md": "42", "sources": ["a"], "coverage": "full",
        "output_path": "vault/outputs/q.md", "session_id": "s1",
    }
    with patch("pkms.coordinator.run_query", return_value=query_result) as mock_query, \
         patch("pkms.coordinator.init_db"):
        result = handle_query("What is life?", "alice", str(vault), CFG)

    mock_query.assert_called_once()
    assert result["answer_md"] == "42"


def test_handle_query_passes_prior_context(tmp_path):
    vault = tmp_path
    (vault / "vault").mkdir()
    query_result = {
        "answer_md": "ans", "sources": [], "coverage": "none",
        "output_path": "vault/outputs/q.md", "session_id": "s1",
    }
    with patch("pkms.coordinator.run_query", return_value=query_result) as mock_query, \
         patch("pkms.coordinator.init_db"):
        handle_query("Q?", "bob", str(vault), CFG, prior_context=["prior mem"])

    call_kwargs = mock_query.call_args[1]
    assert call_kwargs["prior_context"] == ["prior mem"]


# ── handle_lint ───────────────────────────────────────────────────────────────

def test_handle_lint_takes_no_lock(tmp_path):
    # Linter is read-only — must run without touching the lock store (pkms_seq_lint.md)
    vault = tmp_path
    (vault / "vault").mkdir()
    lint_result = {"total_issues": 0, "issues_by_type": {}, "report_path": "out.md", "report_md": ""}

    with patch("pkms.lock.acquire") as mock_acq, \
         patch("pkms.lock.release") as mock_rel, \
         patch("pkms.coordinator.run_lint", return_value=lint_result) as mock_lint, \
         patch("pkms.coordinator.init_db"):
        result = handle_lint(vault_root=str(vault), config=CFG)

    mock_acq.assert_not_called()
    mock_rel.assert_not_called()
    mock_lint.assert_called_once()
    assert result["total_issues"] == 0


def test_handle_lint_runs_while_wiki_lock_held(tmp_path):
    # Lint must succeed even when a compiler holds the wiki lock
    from pkms.lock import acquire as real_acquire, init_locks_db as real_init

    vault = tmp_path
    (vault / "vault").mkdir()
    locks_db = str(tmp_path / "locks.db")
    real_init(locks_db)
    real_acquire("wiki", "compiler", locks_db, CFG)  # simulate running compile

    lint_result = {"total_issues": 2, "issues_by_type": {}, "report_path": "out.md", "report_md": ""}
    with patch("pkms.coordinator.run_lint", return_value=lint_result), \
         patch("pkms.coordinator.init_db"):
        result = handle_lint(vault_root=str(vault), config=CFG)

    assert result["total_issues"] == 2


# ── handle_compile: scope validation & post_lint ─────────────────────────────

def test_handle_compile_rejects_unknown_scope_string(tmp_path):
    with pytest.raises(ValueError, match="Unknown compile scope"):
        handle_compile(scope="bogus", vault_root=str(tmp_path), config=CFG)


def test_handle_compile_post_lint_triggered_after_release(tmp_path):
    vault = tmp_path
    (vault / "vault").mkdir()
    cfg = {**CFG, "compile": {**CFG["compile"], "post_lint": True}}
    compile_result = {"status": "DONE", "n_articles": 2, "articles": []}
    order = []

    with patch("pkms.lock.init_locks_db"), \
         patch("pkms.lock.acquire", return_value="tok"), \
         patch("pkms.lock.release", side_effect=lambda *a: order.append("release")), \
         patch("pkms.coordinator.run_compile", return_value=compile_result), \
         patch("pkms.coordinator.handle_lint", side_effect=lambda **kw: order.append("lint")) , \
         patch("pkms.coordinator.init_db"):
        handle_compile(scope="all", vault_root=str(vault), config=cfg)

    assert order == ["release", "lint"]


def test_handle_compile_post_lint_skipped_when_no_articles(tmp_path):
    vault = tmp_path
    (vault / "vault").mkdir()
    cfg = {**CFG, "compile": {**CFG["compile"], "post_lint": True}}
    compile_result = {"status": "DONE", "n_articles": 0, "articles": []}

    with patch("pkms.lock.init_locks_db"), \
         patch("pkms.lock.acquire", return_value="tok"), \
         patch("pkms.lock.release"), \
         patch("pkms.coordinator.run_compile", return_value=compile_result), \
         patch("pkms.coordinator.handle_lint") as mock_lint, \
         patch("pkms.coordinator.init_db"):
        handle_compile(scope="all", vault_root=str(vault), config=cfg)

    mock_lint.assert_not_called()


# ── CLI main ──────────────────────────────────────────────────────────────────

def test_main_ingest_verb(tmp_path, capsys):
    vault = tmp_path
    (vault / "vault").mkdir()
    with patch("pkms.coordinator._VAULT_ROOT", vault), \
         patch("pkms.coordinator.load_config", return_value=CFG), \
         patch("pkms.coordinator.handle_ingest", return_value=INGEST_RESULT) as mock_hi:
        from pkms.coordinator import main
        sys.argv = ["pkms", "ingest", "/path/doc.pdf"]
        main()

    mock_hi.assert_called_once_with("/path/doc.pdf", str(vault), CFG, project="default", force=False)
    captured = capsys.readouterr()
    assert "DONE" in captured.out


def test_main_compile_verb(tmp_path, capsys):
    vault = tmp_path
    (vault / "vault").mkdir()
    compile_result = {"status": "DONE", "n_articles": 1}
    with patch("pkms.coordinator._VAULT_ROOT", vault), \
         patch("pkms.coordinator.load_config", return_value=CFG), \
         patch("pkms.coordinator.handle_compile", return_value=compile_result) as mock_hc:
        from pkms.coordinator import main
        sys.argv = ["pkms", "compile"]
        main()

    mock_hc.assert_called_once_with(scope={"type": "full"}, vault_root=str(vault), config=CFG, project="default")
    assert "1" in capsys.readouterr().out


def test_main_reindex_verb(tmp_path, capsys):
    vault = tmp_path
    (vault / "vault").mkdir()
    res = {"status": "DONE", "project": "p", "reingested": 3, "articles_rebuilt": 3}
    with patch("pkms.coordinator._VAULT_ROOT", vault), \
         patch("pkms.coordinator.load_config", return_value=CFG), \
         patch("pkms.coordinator.ensure_collections"), \
         patch("pkms.coordinator.handle_reindex", return_value=res) as mock_rx:
        from pkms.coordinator import main
        sys.argv = ["pkms", "reindex", "--project", "p"]
        main()
    mock_rx.assert_called_once_with("p", str(vault), CFG)
    assert "3 re-ingested" in capsys.readouterr().out


def test_main_query_verb(tmp_path, capsys):
    vault = tmp_path
    (vault / "vault").mkdir()
    q_result = {
        "answer_md": "The answer.", "sources": ["a.md"], "coverage": "full",
        "output_path": "vault/outputs/q.md", "session_id": "s1",
    }
    with patch("pkms.coordinator._VAULT_ROOT", vault), \
         patch("pkms.coordinator.load_config", return_value=CFG), \
         patch("pkms.coordinator.handle_query", return_value=q_result) as mock_hq:
        from pkms.coordinator import main
        sys.argv = ["pkms", "query", "What is AI?"]
        main()

    mock_hq.assert_called_once()
    out = capsys.readouterr().out
    assert "The answer." in out
    assert "full" in out


def test_main_lint_verb(tmp_path, capsys):
    vault = tmp_path
    (vault / "vault").mkdir()
    lint_result = {"total_issues": 2, "issues_by_type": {}, "report_path": "vault/outputs/lint.md", "report_md": ""}
    with patch("pkms.coordinator._VAULT_ROOT", vault), \
         patch("pkms.coordinator.load_config", return_value=CFG), \
         patch("pkms.coordinator.handle_lint", return_value=lint_result):
        from pkms.coordinator import main
        sys.argv = ["pkms", "lint"]
        main()

    assert "2" in capsys.readouterr().out
