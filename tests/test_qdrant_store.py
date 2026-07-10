import uuid
import pytest
from qdrant_client import QdrantClient

from pkms.qdrant_store import (
    delete_by_ids,
    delete_by_path,
    ensure_collections,
    point_id,
    scroll,
    search,
    upsert,
    upsert_batch,
)


CONFIG = {
    "qdrant": {
        "host": "localhost",
        "port": 6333,
        "collections": {"raw": "raw", "wiki": "wiki", "outputs": "outputs"},
    },
    "embedding": {"dimension": 4},  # tiny dim for in-memory tests
}


@pytest.fixture()
def config(monkeypatch):
    """Patch _client to return an in-memory QdrantClient."""
    mem_client = QdrantClient(":memory:")
    monkeypatch.setattr("pkms.qdrant_store._client", lambda cfg: mem_client)
    return CONFIG


# ── point_id ──────────────────────────────────────────────────────────────────

def test_point_id_is_deterministic():
    a = point_id("vault/raw/doc.pdf", 3)
    b = point_id("vault/raw/doc.pdf", 3)
    assert a == b


def test_point_id_differs_by_chunk():
    assert point_id("vault/raw/doc.pdf", 0) != point_id("vault/raw/doc.pdf", 1)


def test_point_id_is_valid_uuid():
    pid = point_id("vault/raw/doc.pdf", 0)
    uuid.UUID(pid)  # must not raise


# ── ensure_collections ────────────────────────────────────────────────────────

def test_ensure_collections_creates_all(config):
    ensure_collections(config)
    from pkms.qdrant_store import _client
    names = {c.name for c in _client(config).get_collections().collections}
    assert {"raw", "wiki", "outputs"}.issubset(names)


def test_ensure_collections_idempotent(config):
    ensure_collections(config)
    ensure_collections(config)  # must not raise


# ── upsert / scroll ───────────────────────────────────────────────────────────

def test_upsert_and_scroll(config):
    ensure_collections(config)
    pid = point_id("vault/raw/a.pdf", 0)
    vec = [0.1, 0.2, 0.3, 0.4]
    payload = {"path": "vault/raw/a.pdf", "chunk_index": 0}
    upsert("raw", pid, vec, payload, config)

    results = scroll("raw", "vault/raw/a.pdf", config)
    assert len(results) == 1
    assert results[0]["payload"]["path"] == "vault/raw/a.pdf"


def test_scroll_returns_empty_for_unknown_path(config):
    ensure_collections(config)
    assert scroll("raw", "vault/raw/missing.pdf", config) == []


def test_upsert_is_idempotent(config):
    ensure_collections(config)
    pid = point_id("vault/raw/a.pdf", 0)
    vec = [0.1, 0.2, 0.3, 0.4]
    payload = {"path": "vault/raw/a.pdf", "chunk_index": 0, "title": "v1"}
    upsert("raw", pid, vec, payload, config)
    payload2 = {**payload, "title": "v2"}
    upsert("raw", pid, vec, payload2, config)
    results = scroll("raw", "vault/raw/a.pdf", config)
    assert len(results) == 1
    assert results[0]["payload"]["title"] == "v2"


# ── search ────────────────────────────────────────────────────────────────────

def test_search_returns_nearest(config):
    ensure_collections(config)
    for i, vec in enumerate([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]]):
        pid = point_id("vault/raw/doc.pdf", i)
        upsert("raw", pid, [float(v) for v in vec], {"path": "vault/raw/doc.pdf", "chunk_index": i}, config)

    results = search("raw", [1.0, 0.0, 0.0, 0.0], top_k=1, config=config)
    assert len(results) == 1
    assert results[0]["payload"]["chunk_index"] == 0


def test_search_respects_top_k(config):
    ensure_collections(config)
    for i in range(5):
        pid = point_id("vault/raw/doc.pdf", i)
        upsert("raw", pid, [float(i), 0.0, 0.0, 0.0], {"path": "vault/raw/doc.pdf"}, config)
    results = search("raw", [1.0, 0.0, 0.0, 0.0], top_k=3, config=config)
    assert len(results) == 3


# ── delete_by_ids ─────────────────────────────────────────────────────────────

def test_delete_by_ids_removes_points(config):
    ensure_collections(config)
    pid = point_id("vault/raw/a.pdf", 0)
    upsert("raw", pid, [0.1, 0.2, 0.3, 0.4], {"path": "vault/raw/a.pdf"}, config)
    delete_by_ids("raw", [pid], config)
    assert scroll("raw", "vault/raw/a.pdf", config) == []


def test_delete_by_ids_empty_list_is_noop(config):
    ensure_collections(config)
    delete_by_ids("raw", [], config)  # must not raise


# ── upsert_batch ──────────────────────────────────────────────────────────────

def test_upsert_batch_writes_all_points(config):
    ensure_collections(config)
    points = [
        (point_id("vault/raw/b.pdf", i), [0.1 * i, 0.2, 0.3, 0.4],
         {"path": "vault/raw/b.pdf", "chunk_index": i})
        for i in range(3)
    ]
    upsert_batch("raw", points, config)
    results = scroll("raw", "vault/raw/b.pdf", config)
    assert len(results) == 3
    assert {r["payload"]["chunk_index"] for r in results} == {0, 1, 2}


def test_upsert_batch_empty_is_noop(config):
    ensure_collections(config)
    upsert_batch("raw", [], config)  # must not raise


# ── delete_by_path (filtered delete) ──────────────────────────────────────────

def test_delete_by_path_removes_only_that_path(config):
    ensure_collections(config)
    for path in ("vault/raw/x.pdf", "vault/raw/y.pdf"):
        for i in range(2):
            upsert("raw", point_id(path, i), [0.1, 0.2, 0.3, 0.4],
                   {"path": path, "chunk_index": i}, config)

    n = delete_by_path("raw", "vault/raw/x.pdf", config)

    assert n == 2                                             # count contract preserved
    assert scroll("raw", "vault/raw/x.pdf", config) == []     # gone
    assert len(scroll("raw", "vault/raw/y.pdf", config)) == 2 # untouched


def test_delete_by_path_unknown_path_returns_zero(config):
    ensure_collections(config)
    assert delete_by_path("raw", "vault/raw/ghost.pdf", config) == 0
