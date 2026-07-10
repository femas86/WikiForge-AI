from pathlib import Path
from unittest.mock import patch

import pytest

import pkms.qdrant_store as qs
from pkms.compiler import _strip_dangling_links
from pkms.coordinator import handle_remove_doc
from pkms.db import (
    delete_article_sources,
    delete_file,
    get_article_sources,
    get_articles_for_raw,
    get_file,
    init_db,
    upsert_article_source,
    upsert_file,
)

CFG = {
    "qdrant": {"host": "localhost", "port": 6333,
               "collections": {"raw": "raw", "wiki": "wiki", "outputs": "outputs"}},
    "embedding": {"dimension": 4},
    "chunking": {"max_tokens": 200},
    "compile": {"crosslink_threshold": 0.82, "post_lint": False},
    "lock": {"ttl_seconds": 300, "max_wait_seconds": 1},
}


# ── db delete helpers ─────────────────────────────────────────────────────────

def _seed(db):
    upsert_file(db, path="vault/p/raw/foo.md", hash="h1", qdrant_ids=["r1"],
                collection="raw", indexed_at="t", project="p")
    upsert_file(db, path="vault/p/wiki/articles/foo.md", hash="h2", qdrant_ids=["w1"],
                collection="wiki", indexed_at="t", project="p")
    upsert_article_source(db, wiki_path="vault/p/wiki/articles/foo.md",
                          raw_path="vault/p/raw/foo.md", compile_hash="h1", compiled_at="t")


def test_get_articles_for_raw(tmp_path):
    db = str(tmp_path / ".search-index"); init_db(db); _seed(db)
    assert get_articles_for_raw(db, "vault/p/raw/foo.md") == ["vault/p/wiki/articles/foo.md"]
    assert get_articles_for_raw(db, "vault/p/raw/missing.md") == []


def test_delete_file(tmp_path):
    db = str(tmp_path / ".search-index"); init_db(db); _seed(db)
    delete_file(db, "vault/p/raw/foo.md")
    assert get_file(db, "vault/p/raw/foo.md") is None
    assert get_file(db, "vault/p/wiki/articles/foo.md") is not None  # only the one row


def test_delete_article_sources_by_wiki(tmp_path):
    db = str(tmp_path / ".search-index"); init_db(db); _seed(db)
    delete_article_sources(db, wiki_path="vault/p/wiki/articles/foo.md")
    assert get_article_sources(db, "vault/p/wiki/articles/foo.md") == []


def test_delete_article_sources_requires_a_key(tmp_path):
    db = str(tmp_path / ".search-index"); init_db(db)
    with pytest.raises(ValueError):
        delete_article_sources(db)


# ── qdrant delete_by_path ─────────────────────────────────────────────────────

def test_delete_by_path_counts_then_filtered_deletes():
    # One server-side filtered delete (no scroll → no 10k page truncation);
    # count read first to preserve the return contract.
    from unittest.mock import MagicMock
    client = MagicMock()
    client.count.return_value.count = 2
    with patch.object(qs, "_client", return_value=client):
        n = qs.delete_by_path("raw", "vault/p/raw/foo.md", CFG)
    assert n == 2
    client.count.assert_called_once()
    client.delete.assert_called_once()
    selector = client.delete.call_args.kwargs["points_selector"]
    cond = selector.filter.must[0]
    assert cond.key == "path" and cond.match.value == "vault/p/raw/foo.md"


def test_delete_by_path_skips_delete_when_no_points():
    from unittest.mock import MagicMock
    client = MagicMock()
    client.count.return_value.count = 0
    with patch.object(qs, "_client", return_value=client):
        assert qs.delete_by_path("raw", "vault/p/raw/ghost.md", CFG) == 0
    client.delete.assert_not_called()


# ── dangling-link strip ───────────────────────────────────────────────────────

def test_strip_dangling_links(tmp_path):
    articles = tmp_path / "vault" / "p" / "wiki" / "articles"
    articles.mkdir(parents=True)
    sibling = articles / "bar.md"
    sibling.write_text(
        "# Bar\n\nSee the [[foo]] concept and [[baz]] too.\n\n"
        "## See also\n\n- [[foo]] — the removed doc\n- [[baz]] — still here\n",
        encoding="utf-8",
    )
    empty = articles / "solo.md"
    empty.write_text("# Solo\n\n## See also\n\n- [[foo]] — only link\n", encoding="utf-8")

    with patch("pkms.compiler.guard_write"):
        n = _strip_dangling_links(tmp_path / "vault", "p", {"foo"}, CFG)

    assert n == 2
    bar = sibling.read_text()
    assert "[[foo]]" not in bar                 # inline degraded + See-also bullet dropped
    assert "the [[baz]] too" in bar or "[[baz]]" in bar  # unrelated link preserved
    assert "- [[baz]] — still here" in bar
    solo = empty.read_text()
    assert "See also" not in solo               # emptied section header removed
    assert "[[foo]]" not in solo


# ── handle_remove_doc: sole-source article ────────────────────────────────────

def _fs_and_db(tmp_path, sources):
    """Create vault files + db rows for one wiki article with the given raw sources."""
    vault = tmp_path / "vault"
    (vault / "p" / "raw").mkdir(parents=True)
    (vault / "p" / "wiki" / "articles").mkdir(parents=True)
    db = str(vault / ".search-index"); init_db(db)
    wiki_path = "vault/p/wiki/articles/topic.md"
    (vault / "p" / "wiki" / "articles" / "topic.md").write_text("# Topic\n", encoding="utf-8")
    upsert_file(db, path=wiki_path, hash="hw", qdrant_ids=["w1"],
                collection="wiki", indexed_at="t", project="p")
    for src in sources:
        (tmp_path / src).write_text("raw\n", encoding="utf-8")
        upsert_file(db, path=src, hash="hr", qdrant_ids=["r1"],
                    collection="raw", indexed_at="t", project="p")
        upsert_article_source(db, wiki_path=wiki_path, raw_path=src,
                              compile_hash="hr", compiled_at="t")
    return str(tmp_path), db, wiki_path


def _patches():
    return (
        patch("pkms.lock.init_locks_db"),
        patch("pkms.lock.acquire", return_value="tok"),
        patch("pkms.lock.release"),
        patch("pkms.coordinator.delete_by_path", return_value=3),
        patch("pkms.coordinator._git_commit_paths"),
        patch("pkms.coordinator._rebuild_index"),
    )


def test_remove_sole_source_cleans_all_surfaces(tmp_path):
    vault_root, db, wiki_path = _fs_and_db(tmp_path, ["vault/p/raw/topic.md"])
    # a sibling article linking the doomed one
    sib = Path(vault_root) / "vault" / "p" / "wiki" / "articles" / "sib.md"
    sib.write_text("# Sib\n\n## See also\n\n- [[topic]] — bye\n", encoding="utf-8")

    p_locks, p_acq, p_rel, p_qd, p_git, p_idx = _patches()
    with p_locks, p_acq, p_rel, p_qd as mock_qd, p_git, p_idx, \
         patch("pkms.coordinator.handle_compile") as mock_compile, \
         patch("pkms.coordinator.handle_lint") as mock_lint:
        result = handle_remove_doc("vault/p/raw/topic.md", vault_root, CFG, project="p")

    # filesystem
    assert not (Path(vault_root) / "vault/p/raw/topic.md").exists()
    assert not (Path(vault_root) / "vault/p/wiki/articles/topic.md").exists()
    # db
    assert get_file(db, "vault/p/raw/topic.md") is None
    assert get_file(db, wiki_path) is None
    assert get_article_sources(db, wiki_path) == []
    # qdrant: raw + wiki both purged by path
    assert mock_qd.call_count == 2
    # sibling dangling link stripped
    assert "[[topic]]" not in sib.read_text()
    # sole-source ⇒ lint (not recompile)
    mock_compile.assert_not_called()
    mock_lint.assert_called_once()
    assert result["recompiled"] is False
    assert result["articles_removed"] == ["topic"]


def test_remove_multi_source_triggers_recompile(tmp_path):
    vault_root, db, wiki_path = _fs_and_db(
        tmp_path, ["vault/p/raw/a.md", "vault/p/raw/b.md"])

    p_locks, p_acq, p_rel, p_qd, p_git, p_idx = _patches()
    with p_locks, p_acq, p_rel, p_qd, p_git, p_idx, \
         patch("pkms.coordinator.handle_compile") as mock_compile, \
         patch("pkms.coordinator.handle_lint") as mock_lint:
        result = handle_remove_doc("vault/p/raw/a.md", vault_root, CFG, project="p")

    # removed raw gone; remaining raw kept for the rebuild
    assert get_file(db, "vault/p/raw/a.md") is None
    assert get_file(db, "vault/p/raw/b.md") is not None
    # article torn down; its sources cleared so remaining raw is 'uncompiled'
    assert get_file(db, wiki_path) is None
    assert get_article_sources(db, wiki_path) == []
    # remaining source ⇒ recompile (not a bare lint)
    mock_compile.assert_called_once_with(
        scope="all", vault_root=vault_root, config=CFG, project="p")
    mock_lint.assert_not_called()
    assert result["recompiled"] is True


def test_remove_rejects_project_mismatch(tmp_path):
    vault_root, _, _ = _fs_and_db(tmp_path, ["vault/p/raw/topic.md"])
    with pytest.raises(ValueError, match="Project mismatch"):
        handle_remove_doc("vault/p/raw/topic.md", vault_root, CFG, project="other")


def test_remove_leaves_no_phantom_graph_node(tmp_path):
    """The wiki graph is recomputed from files on the fly, so a removed article's
    node must vanish AND its inbound [[links]] must be stripped — otherwise the
    dead slug resurfaces as a phantom 'concept' node (web._wiki_link_graph)."""
    from pkms.web import _wiki_link_graph

    vault_root, _, _ = _fs_and_db(tmp_path, ["vault/p/raw/topic.md"])
    articles = Path(vault_root) / "vault" / "p" / "wiki" / "articles"
    (articles / "topic.md").write_text(
        "---\ntitle: Topic\ntags: [ml]\n---\n# Topic\n", encoding="utf-8")
    (articles / "sib.md").write_text(
        "---\ntitle: Sib\n---\n# Sib\n\nSee [[topic]].\n\n## See also\n\n- [[topic]] — bye\n",
        encoding="utf-8")

    p_locks, p_acq, p_rel, p_qd, p_git, p_idx = _patches()
    with p_locks, p_acq, p_rel, p_qd, p_git, p_idx, \
         patch("pkms.coordinator.handle_compile"), \
         patch("pkms.coordinator.handle_lint"):
        handle_remove_doc("vault/p/raw/topic.md", vault_root, CFG, project="p")

    graph = _wiki_link_graph(Path(vault_root) / "vault", "p")
    ids = {n["id"] for n in graph["nodes"]}
    assert "topic" not in ids                    # article node gone
    assert "concept:topic" not in ids            # no phantom concept from dangling links
    assert not any(e["target"] in {"topic", "concept:topic"} for e in graph["links"])


def test_main_remove_verb(tmp_path, capsys):
    import sys
    (tmp_path / "vault").mkdir()
    removed = {"status": "REMOVED", "raw_path": "vault/p/raw/x.pdf",
               "raw_points_deleted": 5, "articles_removed": ["x"], "recompiled": False}
    with patch("pkms.coordinator._VAULT_ROOT", tmp_path), \
         patch("pkms.coordinator.load_config", return_value=CFG), \
         patch("pkms.coordinator.ensure_collections"), \
         patch("pkms.coordinator.handle_remove_doc", return_value=removed) as mock_rm:
        from pkms.coordinator import main
        sys.argv = ["pkms", "remove", "vault/p/raw/x.pdf", "--project", "p"]
        main()
    mock_rm.assert_called_once_with("vault/p/raw/x.pdf", str(tmp_path), CFG, project="p")
    assert "REMOVED" in capsys.readouterr().out
