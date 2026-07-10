import uuid
from functools import lru_cache
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    FilterSelector,
    MatchValue,
    PointIdsList,
    PointStruct,
    VectorParams,
)


@lru_cache(maxsize=None)
def _cached_client(host: str, port: int) -> QdrantClient:
    # One client (with its httpx connection pool) per endpoint for the process
    # lifetime — building a fresh client per operation paid a TCP handshake on
    # every upsert/scroll/search. REST-mode QdrantClient is thread-safe.
    return QdrantClient(host=host, port=port)


def _client(config: dict[str, Any]) -> QdrantClient:
    return _cached_client(config["qdrant"]["host"], config["qdrant"]["port"])


def point_id(path: str, chunk_index: int) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"{path}:{chunk_index}"))


def ensure_collections(config: dict[str, Any]) -> None:
    """Create all three collections (with a project payload index) if missing."""
    client = _client(config)
    dim = config["embedding"]["dimension"]
    collections = config["qdrant"]["collections"]
    existing = {c.name for c in client.get_collections().collections}
    for name in collections.values():
        if name not in existing:
            client.create_collection(
                collection_name=name,
                vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
            )
            client.create_payload_index(
                collection_name=name,
                field_name="project",
                field_schema="keyword",
            )


def upsert_batch(
    collection: str,
    points: list[tuple[str, list[float], dict[str, Any]]],
    config: dict[str, Any],
) -> None:
    """Upsert many (point_id, vector, payload) triples in ONE request.

    A whole document's chunks land in a single call (atomic per request on
    Qdrant's side) instead of one round trip per chunk.
    """
    if not points:
        return
    client = _client(config)
    client.upsert(
        collection_name=collection,
        points=[PointStruct(id=pid, vector=vec, payload=pl) for pid, vec, pl in points],
    )


def upsert(
    collection: str,
    point_id_: str,
    vector: list[float],
    payload: dict[str, Any],
    config: dict[str, Any],
) -> None:
    upsert_batch(collection, [(point_id_, vector, payload)], config)


def scroll(
    collection: str,
    path_filter: str,
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    """Return all points whose payload.path matches path_filter."""
    client = _client(config)
    results, _ = client.scroll(
        collection_name=collection,
        scroll_filter=Filter(
            must=[FieldCondition(key="path", match=MatchValue(value=path_filter))]
        ),
        with_payload=True,
        with_vectors=False,
        limit=10_000,
    )
    return [{"id": str(p.id), "payload": p.payload} for p in results]


def search(
    collection: str,
    vector: list[float],
    top_k: int,
    config: dict[str, Any],
    project: str | None = None,
) -> list[dict[str, Any]]:
    """Vector search; project scopes results to one project's documents."""
    client = _client(config)
    query_filter = None
    if project is not None:
        query_filter = Filter(
            must=[FieldCondition(key="project", match=MatchValue(value=project))]
        )
    response = client.query_points(
        collection_name=collection,
        query=vector,
        limit=top_k,
        query_filter=query_filter,
        with_payload=True,
    )
    return [{"id": str(h.id), "score": h.score, "payload": h.payload} for h in response.points]


def delete_by_ids(
    collection: str,
    ids: list[str],
    config: dict[str, Any],
) -> None:
    if not ids:
        return
    client = _client(config)
    client.delete(
        collection_name=collection,
        points_selector=PointIdsList(points=ids),
    )


def delete_by_path(
    collection: str,
    path: str,
    config: dict[str, Any],
) -> int:
    """Delete every point whose payload.path == path. Returns count deleted.

    Used to un-ingest a document without needing its stored point ids, so it is
    robust even if the .search-index row (which holds qdrant_ids) is already
    gone or out of sync. A single server-side filtered delete: no scroll (whose
    10k page limit silently orphaned points on huge documents) and no payload
    transfer; the count is read first to preserve the return contract.
    """
    client = _client(config)
    flt = Filter(must=[FieldCondition(key="path", match=MatchValue(value=path))])
    n = client.count(collection_name=collection, count_filter=flt, exact=True).count
    if n:
        client.delete(collection_name=collection, points_selector=FilterSelector(filter=flt))
    return n
