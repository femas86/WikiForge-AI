import pytest
from pkms.db import (
    init_db,
    upsert_file,
    get_file,
    get_missing_files,
    upsert_article_source,
    get_article_sources,
    get_stale_articles,
    set_config,
    get_config,
)


@pytest.fixture()
def db(tmp_path):
    path = str(tmp_path / "search-index")
    init_db(path)
    return path


# ── init_db ───────────────────────────────────────────────────────────────────

def test_init_db_idempotent(tmp_path):
    path = str(tmp_path / "index")
    init_db(path)
    init_db(path)  # must not raise


# ── files table ───────────────────────────────────────────────────────────────

def test_upsert_and_get_file(db):
    upsert_file(db, path="vault/raw/doc.pdf", hash="sha256:abc",
                qdrant_ids=["id1", "id2"], collection="raw",
                indexed_at="2026-06-11T10:00:00Z")
    row = get_file(db, "vault/raw/doc.pdf")
    assert row["path"] == "vault/raw/doc.pdf"
    assert row["hash"] == "sha256:abc"
    assert row["qdrant_ids"] == ["id1", "id2"]
    assert row["collection"] == "raw"


def test_upsert_file_updates_existing(db):
    upsert_file(db, path="vault/raw/doc.pdf", hash="sha256:old",
                qdrant_ids=["x"], collection="raw", indexed_at="2026-06-11T09:00:00Z")
    upsert_file(db, path="vault/raw/doc.pdf", hash="sha256:new",
                qdrant_ids=["y", "z"], collection="raw", indexed_at="2026-06-11T10:00:00Z")
    row = get_file(db, "vault/raw/doc.pdf")
    assert row["hash"] == "sha256:new"
    assert row["qdrant_ids"] == ["y", "z"]


def test_get_file_returns_none_for_missing(db):
    assert get_file(db, "vault/raw/nonexistent.pdf") is None


def test_content_hash_roundtrip(db):
    upsert_file(db, path="vault/raw/page.html", hash="sha256:bytes",
                content_hash="sha256:text", qdrant_ids=["id1"],
                collection="raw", indexed_at="2026-07-03T10:00:00Z")
    row = get_file(db, "vault/raw/page.html")
    assert row["content_hash"] == "sha256:text"


def test_content_hash_defaults_empty(db):
    # a writer that omits content_hash (e.g. the compiler for wiki rows) → ''
    upsert_file(db, path="vault/raw/doc.pdf", hash="sha256:b",
                qdrant_ids=[], collection="raw", indexed_at="2026-07-03T10:00:00Z")
    assert get_file(db, "vault/raw/doc.pdf")["content_hash"] == ""


def test_init_db_migrates_legacy_files_table(tmp_path):
    # A pre-content_hash DB (old schema) must gain the column without data loss.
    import sqlite3
    path = str(tmp_path / "legacy-index")
    conn = sqlite3.connect(path)
    conn.execute("""CREATE TABLE files (
        path TEXT PRIMARY KEY, project TEXT NOT NULL DEFAULT 'default',
        hash TEXT NOT NULL, qdrant_ids TEXT NOT NULL,
        collection TEXT NOT NULL, indexed_at TEXT NOT NULL)""")
    conn.execute("INSERT INTO files VALUES ('vault/raw/x.pdf','default','sha256:h','[]','raw','t')")
    conn.commit(); conn.close()

    init_db(path)  # migration adds content_hash
    row = get_file(path, "vault/raw/x.pdf")
    assert row is not None
    assert row["content_hash"] == ""     # legacy row → empty → re-ingests once


def test_get_missing_files(db):
    upsert_file(db, path="vault/raw/a.pdf", hash="sha256:1",
                qdrant_ids=[], collection="raw", indexed_at="2026-06-11T10:00:00Z")
    upsert_file(db, path="vault/raw/b.pdf", hash="sha256:2",
                qdrant_ids=[], collection="raw", indexed_at="2026-06-11T10:00:00Z")
    missing = get_missing_files(db, existing_paths={"vault/raw/a.pdf"})
    assert missing == ["vault/raw/b.pdf"]


def test_get_missing_files_all_present(db):
    upsert_file(db, path="vault/raw/a.pdf", hash="sha256:1",
                qdrant_ids=[], collection="raw", indexed_at="2026-06-11T10:00:00Z")
    assert get_missing_files(db, existing_paths={"vault/raw/a.pdf"}) == []


# ── article_sources table ─────────────────────────────────────────────────────

def _seed_files(db):
    upsert_file(db, path="vault/wiki/articles/t.md", hash="sha256:w",
                qdrant_ids=[], collection="wiki", indexed_at="2026-06-11T10:00:00Z")
    upsert_file(db, path="vault/raw/src.pdf", hash="sha256:r",
                qdrant_ids=[], collection="raw", indexed_at="2026-06-11T10:00:00Z")


def test_upsert_and_get_article_source(db):
    _seed_files(db)
    upsert_article_source(db, wiki_path="vault/wiki/articles/t.md",
                          raw_path="vault/raw/src.pdf",
                          compile_hash="sha256:r",
                          compiled_at="2026-06-11T10:00:00Z")
    rows = get_article_sources(db, "vault/wiki/articles/t.md")
    assert len(rows) == 1
    assert rows[0]["compile_hash"] == "sha256:r"


def test_upsert_article_source_updates(db):
    _seed_files(db)
    upsert_article_source(db, wiki_path="vault/wiki/articles/t.md",
                          raw_path="vault/raw/src.pdf",
                          compile_hash="sha256:old",
                          compiled_at="2026-06-11T09:00:00Z")
    upsert_article_source(db, wiki_path="vault/wiki/articles/t.md",
                          raw_path="vault/raw/src.pdf",
                          compile_hash="sha256:new",
                          compiled_at="2026-06-11T10:00:00Z")
    rows = get_article_sources(db, "vault/wiki/articles/t.md")
    assert rows[0]["compile_hash"] == "sha256:new"


def test_get_article_sources_empty(db):
    assert get_article_sources(db, "vault/wiki/articles/missing.md") == []


# ── drift detection ───────────────────────────────────────────────────────────

def test_get_stale_articles_detects_drift(db):
    upsert_file(db, path="vault/wiki/articles/t.md", hash="sha256:w",
                qdrant_ids=[], collection="wiki", indexed_at="2026-06-11T10:00:00Z")
    upsert_file(db, path="vault/raw/src.pdf", hash="sha256:NEW",  # changed
                qdrant_ids=[], collection="raw", indexed_at="2026-06-11T11:00:00Z")
    upsert_article_source(db, wiki_path="vault/wiki/articles/t.md",
                          raw_path="vault/raw/src.pdf",
                          compile_hash="sha256:OLD",  # stale
                          compiled_at="2026-06-11T10:00:00Z")
    stale = get_stale_articles(db)
    assert len(stale) == 1
    assert stale[0]["wiki_path"] == "vault/wiki/articles/t.md"
    assert stale[0]["stale_hash"] == "sha256:OLD"
    assert stale[0]["current_hash"] == "sha256:NEW"


def test_get_stale_articles_no_drift(db):
    _seed_files(db)
    upsert_article_source(db, wiki_path="vault/wiki/articles/t.md",
                          raw_path="vault/raw/src.pdf",
                          compile_hash="sha256:r",  # matches files.hash
                          compiled_at="2026-06-11T10:00:00Z")
    assert get_stale_articles(db) == []


# ── config table ──────────────────────────────────────────────────────────────

def test_set_and_get_config(db):
    set_config(db, "schema_version", "1")
    assert get_config(db, "schema_version") == "1"


def test_set_config_overwrites(db):
    set_config(db, "key", "v1")
    set_config(db, "key", "v2")
    assert get_config(db, "key") == "v2"


def test_get_config_missing_returns_none(db):
    assert get_config(db, "nonexistent") is None


# ── project column ────────────────────────────────────────────────────────────

def test_upsert_file_stores_project(tmp_path):
    db = str(tmp_path / ".search-index")
    init_db(db)
    upsert_file(db, path="vault/robotics/raw/a.pdf", hash="sha256:a",
                qdrant_ids=["i1"], collection="raw",
                indexed_at="2026-06-12T08:00:00Z", project="robotics")
    row = get_file(db, "vault/robotics/raw/a.pdf")
    assert row["project"] == "robotics"


def test_upsert_file_defaults_to_default_project(tmp_path):
    db = str(tmp_path / ".search-index")
    init_db(db)
    upsert_file(db, path="vault/default/raw/b.pdf", hash="sha256:b",
                qdrant_ids=["i2"], collection="raw",
                indexed_at="2026-06-12T08:00:00Z")
    assert get_file(db, "vault/default/raw/b.pdf")["project"] == "default"


def test_get_stale_articles_filters_by_project(tmp_path):
    from pkms.db import get_stale_articles, upsert_article_source
    db = str(tmp_path / ".search-index")
    init_db(db)
    for proj in ("alpha", "beta"):
        raw = f"vault/{proj}/raw/doc.pdf"
        wiki = f"vault/{proj}/wiki/articles/doc.md"
        upsert_file(db, path=raw, hash="sha256:new", qdrant_ids=["x"],
                    collection="raw", indexed_at="t", project=proj)
        upsert_article_source(db, wiki_path=wiki, raw_path=raw,
                              compile_hash="sha256:old", compiled_at="t")
    assert len(get_stale_articles(db)) == 2
    alpha_only = get_stale_articles(db, project="alpha")
    assert len(alpha_only) == 1
    assert alpha_only[0]["raw_path"] == "vault/alpha/raw/doc.pdf"


def test_init_db_migrates_pre_project_schema(tmp_path):
    import sqlite3
    db = str(tmp_path / ".search-index")
    conn = sqlite3.connect(db)
    conn.executescript("""
        CREATE TABLE files (
            path TEXT PRIMARY KEY, hash TEXT NOT NULL,
            qdrant_ids TEXT NOT NULL, collection TEXT NOT NULL,
            indexed_at TEXT NOT NULL
        );
        INSERT INTO files VALUES ('vault/raw/old.pdf','sha256:o','[]','raw','t');
    """)
    conn.commit()
    conn.close()
    init_db(db)  # must add the project column without losing data
    row = get_file(db, "vault/raw/old.pdf")
    assert row["project"] == "default"


# ── project_members (B4 multi-user) ─────────────────────────────────────────────

def test_member_add_get_role(tmp_path):
    from pkms.db import init_db, add_member, get_member_role
    db = str(tmp_path / "idx"); init_db(db)
    assert get_member_role(db, "demo", "alice") is None      # not a member
    add_member(db, "demo", "alice", "owner", "t0")
    assert get_member_role(db, "demo", "alice") == "owner"


def test_member_role_upsert(tmp_path):
    from pkms.db import init_db, add_member, get_member_role
    db = str(tmp_path / "idx"); init_db(db)
    add_member(db, "demo", "alice", "viewer", "t0")
    add_member(db, "demo", "alice", "editor", "t1")           # same (project,user) → update role
    assert get_member_role(db, "demo", "alice") == "editor"


def test_member_remove(tmp_path):
    from pkms.db import init_db, add_member, remove_member, get_member_role
    db = str(tmp_path / "idx"); init_db(db)
    add_member(db, "demo", "alice", "owner", "t0")
    remove_member(db, "demo", "alice")
    assert get_member_role(db, "demo", "alice") is None


def test_list_members_and_user_projects(tmp_path):
    from pkms.db import init_db, add_member, list_members, list_user_projects
    db = str(tmp_path / "idx"); init_db(db)
    add_member(db, "demo", "alice", "owner", "t0")
    add_member(db, "demo", "bob", "viewer", "t1")
    add_member(db, "other", "alice", "editor", "t2")
    members = list_members(db, "demo")
    assert {m["user_id"] for m in members} == {"alice", "bob"}
    assert next(m for m in members if m["user_id"] == "alice")["role"] == "owner"
    assert list_user_projects(db, "alice") == ["demo", "other"]   # sorted, both projects
    assert list_user_projects(db, "bob") == ["demo"]
