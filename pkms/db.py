import json
import sqlite3
from contextlib import contextmanager
from typing import Any, Iterator


def _connect(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path, timeout=5.0)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=5000")
    conn.row_factory = sqlite3.Row
    return conn


@contextmanager
def connection(db_path: str) -> Iterator[sqlite3.Connection]:
    """One configured connection (WAL + busy_timeout, Row factory) for a block.

    For callers doing several queries in a row: each helper in this module
    otherwise opens its own connection and re-runs the PRAGMAs — fine for a
    one-off, wasteful inside a loop.
    """
    conn = _connect(db_path)
    try:
        yield conn
    finally:
        conn.close()


def init_db(db_path: str) -> None:
    """Create all tables and indexes if they don't exist."""
    conn = _connect(db_path)
    with conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS files (
                path          TEXT PRIMARY KEY,
                project       TEXT NOT NULL DEFAULT 'default',
                hash          TEXT NOT NULL,
                content_hash  TEXT NOT NULL DEFAULT '',
                qdrant_ids    TEXT NOT NULL,
                collection    TEXT NOT NULL,
                indexed_at    TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_files_collection
                ON files(collection);

            CREATE TABLE IF NOT EXISTS article_sources (
                wiki_path     TEXT NOT NULL,
                raw_path      TEXT NOT NULL,
                compile_hash  TEXT NOT NULL,
                compiled_at   TEXT NOT NULL,
                PRIMARY KEY (wiki_path, raw_path),
                FOREIGN KEY (wiki_path) REFERENCES files(path),
                FOREIGN KEY (raw_path)  REFERENCES files(path)
            );

            CREATE INDEX IF NOT EXISTS idx_article_sources_raw
                ON article_sources(raw_path);
            CREATE INDEX IF NOT EXISTS idx_article_sources_wiki
                ON article_sources(wiki_path);

            CREATE TABLE IF NOT EXISTS config (
                key   TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
        """)
        # Migration for pre-project databases: add the column if missing.
        # The project index is created here (not in the script above) so it
        # runs after the column exists on migrated databases.
        cols = {r["name"] for r in conn.execute("PRAGMA table_info(files)")}
        if "project" not in cols:
            conn.execute(
                "ALTER TABLE files ADD COLUMN project TEXT NOT NULL DEFAULT 'default'"
            )
        # content_hash (hash of the EXTRACTED text) lets a re-ingest skip when only
        # the raw bytes drifted (dynamic HTML) but the parsed content is unchanged.
        # Legacy rows get '' → never matches a real hash → they re-ingest once.
        if "content_hash" not in cols:
            conn.execute(
                "ALTER TABLE files ADD COLUMN content_hash TEXT NOT NULL DEFAULT ''"
            )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_files_project_collection"
            " ON files(project, collection)"
        )
    conn.close()


# ── files table ──────────────────────────────────────────────────────────────

def upsert_file(
    db_path: str,
    *,
    path: str,
    hash: str,
    qdrant_ids: list[str],
    collection: str,
    indexed_at: str,
    project: str = "default",
    content_hash: str = "",
) -> None:
    conn = _connect(db_path)
    with conn:
        conn.execute(
            """
            INSERT INTO files (path, project, hash, content_hash, qdrant_ids, collection, indexed_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(path) DO UPDATE SET
                project      = excluded.project,
                hash         = excluded.hash,
                content_hash = excluded.content_hash,
                qdrant_ids   = excluded.qdrant_ids,
                collection   = excluded.collection,
                indexed_at   = excluded.indexed_at
            """,
            (path, project, hash, content_hash, json.dumps(qdrant_ids), collection, indexed_at),
        )
    conn.close()


def get_file(db_path: str, path: str) -> dict[str, Any] | None:
    conn = _connect(db_path)
    row = conn.execute("SELECT * FROM files WHERE path = ?", (path,)).fetchone()
    conn.close()
    if row is None:
        return None
    d = dict(row)
    d["qdrant_ids"] = json.loads(d["qdrant_ids"])
    return d


def list_raw_paths(db_path: str, project: str) -> list[str]:
    """Return every raw document path indexed for a project (for bulk reindex)."""
    conn = _connect(db_path)
    rows = conn.execute(
        "SELECT path FROM files WHERE collection = 'raw' AND project = ? ORDER BY path",
        (project,),
    ).fetchall()
    conn.close()
    return [r["path"] for r in rows]


def get_missing_files(db_path: str, existing_paths: set[str]) -> list[str]:
    """Return all paths in `files` that are not in existing_paths."""
    conn = _connect(db_path)
    rows = conn.execute("SELECT path FROM files").fetchall()
    conn.close()
    return [r["path"] for r in rows if r["path"] not in existing_paths]


# ── article_sources table ─────────────────────────────────────────────────────

def upsert_article_source(
    db_path: str,
    *,
    wiki_path: str,
    raw_path: str,
    compile_hash: str,
    compiled_at: str,
) -> None:
    conn = _connect(db_path)
    with conn:
        conn.execute(
            """
            INSERT INTO article_sources (wiki_path, raw_path, compile_hash, compiled_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(wiki_path, raw_path) DO UPDATE SET
                compile_hash = excluded.compile_hash,
                compiled_at  = excluded.compiled_at
            """,
            (wiki_path, raw_path, compile_hash, compiled_at),
        )
    conn.close()


def get_article_sources(db_path: str, wiki_path: str) -> list[dict[str, Any]]:
    conn = _connect(db_path)
    rows = conn.execute(
        "SELECT * FROM article_sources WHERE wiki_path = ?", (wiki_path,)
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_articles_for_raw(db_path: str, raw_path: str) -> list[str]:
    """Return the wiki_paths of every article compiled (partly) from raw_path."""
    conn = _connect(db_path)
    rows = conn.execute(
        "SELECT DISTINCT wiki_path FROM article_sources WHERE raw_path = ?", (raw_path,)
    ).fetchall()
    conn.close()
    return [r["wiki_path"] for r in rows]


def delete_file(db_path: str, path: str) -> None:
    """Remove a files row (raw or wiki) by its path key. No-op if absent."""
    conn = _connect(db_path)
    with conn:
        conn.execute("DELETE FROM files WHERE path = ?", (path,))
    conn.close()


def delete_article_sources(
    db_path: str,
    *,
    wiki_path: str | None = None,
    raw_path: str | None = None,
) -> None:
    """Delete article_sources rows by wiki_path and/or raw_path.

    At least one of wiki_path/raw_path must be given (guards against wiping the
    whole table). Passing both AND-matches a single (wiki_path, raw_path) row.
    """
    if wiki_path is None and raw_path is None:
        raise ValueError("delete_article_sources requires wiki_path and/or raw_path")
    clauses, params = [], []
    if wiki_path is not None:
        clauses.append("wiki_path = ?")
        params.append(wiki_path)
    if raw_path is not None:
        clauses.append("raw_path = ?")
        params.append(raw_path)
    conn = _connect(db_path)
    with conn:
        conn.execute(
            f"DELETE FROM article_sources WHERE {' AND '.join(clauses)}", tuple(params)
        )
    conn.close()


def get_stale_articles(db_path: str, project: str | None = None) -> list[dict[str, Any]]:
    """Return rows where a source file hash has changed since last compile.

    project=None returns stale articles across all projects.
    """
    conn = _connect(db_path)
    sql = """
        SELECT DISTINCT
               s.wiki_path,
               s.raw_path,
               s.compile_hash  AS stale_hash,
               r.hash          AS current_hash
        FROM   article_sources s
        JOIN   files r ON r.path = s.raw_path
        WHERE  r.hash != s.compile_hash
    """
    params: tuple = ()
    if project is not None:
        sql += " AND r.project = ?"
        params = (project,)
    rows = conn.execute(sql, params).fetchall()
    conn.close()
    return [dict(r) for r in rows]


# ── config table ──────────────────────────────────────────────────────────────

def set_config(db_path: str, key: str, value: str) -> None:
    conn = _connect(db_path)
    with conn:
        conn.execute(
            """
            INSERT INTO config (key, value) VALUES (?, ?)
            ON CONFLICT(key) DO UPDATE SET value = excluded.value
            """,
            (key, value),
        )
    conn.close()


def get_config(db_path: str, key: str) -> str | None:
    conn = _connect(db_path)
    row = conn.execute("SELECT value FROM config WHERE key = ?", (key,)).fetchone()
    conn.close()
    return row["value"] if row else None
