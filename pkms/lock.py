import random
import sqlite3
import time
import uuid
from contextlib import contextmanager
from typing import Any, Iterator


class LockTimeout(Exception):
    pass


# Paths whose locks table has been created this process — the CREATE TABLE is
# a write transaction, pointless to re-run on every acquire (exactly when
# SQLite is most contended).
_initialized: set[str] = set()


def _connect(locks_db: str) -> sqlite3.Connection:
    conn = sqlite3.connect(locks_db, timeout=5.0)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=5000")
    conn.row_factory = sqlite3.Row
    return conn


def init_locks_db(locks_db: str) -> None:
    conn = _connect(locks_db)
    with conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS locks (
                scope        TEXT PRIMARY KEY,
                token        TEXT NOT NULL,
                agent        TEXT NOT NULL,
                acquired_at  TEXT NOT NULL,
                expires_at   TEXT NOT NULL
            )
        """)
    conn.close()


def acquire(scope: str, agent: str, locks_db: str, config: dict[str, Any]) -> str:
    """Acquire a lock on scope for agent. Returns the lock token (UUID4)."""
    ttl = config["lock"]["ttl_seconds"]
    max_wait = config["lock"]["max_wait_seconds"]
    token = str(uuid.uuid4())
    deadline = time.monotonic() + max_wait
    delay = 0.1  # 100ms base

    if locks_db not in _initialized:
        init_locks_db(locks_db)
        _initialized.add(locks_db)

    # One connection for the whole poll loop — re-opening every 100ms hammered
    # SQLite with connect + PRAGMA negotiation exactly under contention.
    conn = _connect(locks_db)
    try:
        while True:
            try:
                with conn:
                    # Evict expired locks before attempting insert
                    conn.execute(
                        "DELETE FROM locks WHERE expires_at < datetime('now')"
                    )
                    conn.execute(
                        """
                        INSERT OR FAIL INTO locks
                            (scope, token, agent, acquired_at, expires_at)
                        VALUES (
                            ?,
                            ?,
                            ?,
                            datetime('now'),
                            datetime('now', ? || ' seconds')
                        )
                        """,
                        (scope, token, agent, str(ttl)),
                    )
                return token
            except sqlite3.IntegrityError:
                pass

            if time.monotonic() >= deadline:
                raise LockTimeout(
                    f"Could not acquire lock on '{scope}' for agent '{agent}' "
                    f"within {max_wait}s"
                )
            # Jitter decorrelates waiting agents (no thundering herd on release)
            time.sleep(min(delay, 5.0) * random.uniform(0.7, 1.3))
            delay = min(delay * 2, 5.0)
    finally:
        conn.close()


@contextmanager
def wiki_lock(project: str, agent: str, locks_db: str, config: dict[str, Any]) -> Iterator[str]:
    """Hold the wiki:{project} lock for a block; always released on exit.

    Replaces the acquire/try/finally-release ritual at every coordinator call
    site (any lock-protocol change now happens here once). Yields the token
    for callers that thread it through (e.g. run_compile). Raises LockTimeout
    if the lock cannot be acquired within lock.max_wait_seconds.
    """
    scope = f"wiki:{project}"
    token = acquire(scope, agent, locks_db, config)
    try:
        yield token
    finally:
        release(scope, token, locks_db)


def release(scope: str, token: str, locks_db: str) -> None:
    """Release a lock. No-op if the token does not match (prevents stale release)."""
    conn = _connect(locks_db)
    with conn:
        conn.execute(
            "DELETE FROM locks WHERE scope = ? AND token = ?",
            (scope, token),
        )
    conn.close()
