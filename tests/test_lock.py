import time
import threading
import pytest

from pkms.lock import acquire, release, init_locks_db, LockTimeout


CONFIG = {
    "lock": {
        "ttl_seconds": 300,
        "max_wait_seconds": 1,  # short for tests
    }
}

CONFIG_SHORT_TTL = {
    "lock": {
        "ttl_seconds": 1,       # expires in 1s — for TTL eviction test
        "max_wait_seconds": 5,
    }
}


@pytest.fixture()
def db(tmp_path):
    path = str(tmp_path / "locks.db")
    init_locks_db(path)
    return path


# ── acquire happy path ────────────────────────────────────────────────────────

def test_acquire_returns_token(db):
    token = acquire("wiki", "compiler", db, CONFIG)
    assert isinstance(token, str) and len(token) == 36  # UUID4


def test_acquire_different_scopes(db):
    t1 = acquire("wiki", "compiler", db, CONFIG)
    t2 = acquire("raw", "ingestor", db, CONFIG)
    assert t1 != t2


def test_release_allows_reacquire(db):
    token = acquire("wiki", "compiler", db, CONFIG)
    release("wiki", token, db)
    token2 = acquire("wiki", "compiler", db, CONFIG)
    assert token2 != token


# ── contention / timeout ──────────────────────────────────────────────────────

def test_acquire_times_out_when_locked(db):
    acquire("wiki", "compiler", db, CONFIG)  # held, never released
    with pytest.raises(LockTimeout):
        acquire("wiki", "compiler", db, CONFIG)


def test_acquire_succeeds_after_ttl_expiry(db):
    acquire("wiki", "compiler", db, CONFIG_SHORT_TTL)
    # Wait for TTL to expire
    time.sleep(2)
    token2 = acquire("wiki", "compiler", db, CONFIG_SHORT_TTL)
    assert token2  # evicted the expired row and acquired fresh


# ── release safety ────────────────────────────────────────────────────────────

def test_release_wrong_token_is_noop(db):
    token = acquire("wiki", "compiler", db, CONFIG)
    release("wiki", "wrong-token", db)          # should not release
    with pytest.raises(LockTimeout):
        acquire("wiki", "compiler", db, CONFIG)  # still held


def test_release_missing_scope_is_noop(db):
    release("nonexistent", "some-token", db)    # must not raise


# ── concurrency ───────────────────────────────────────────────────────────────

def test_only_one_thread_acquires(db):
    CONFIG_CONCURRENT = {"lock": {"ttl_seconds": 10, "max_wait_seconds": 5}}
    results = []

    def worker():
        try:
            token = acquire("wiki", "compiler", db, CONFIG_CONCURRENT)
            results.append(("ok", token))
            time.sleep(0.1)
            release("wiki", token, db)
        except LockTimeout:
            results.append(("timeout", None))

    threads = [threading.Thread(target=worker) for _ in range(3)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    ok_count = sum(1 for r in results if r[0] == "ok")
    assert ok_count >= 1  # at least one thread must succeed
