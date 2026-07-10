"""Shared chunk→embed→upsert step for agents that index content into Qdrant.

Owns the canonical per-chunk payload schema. Ingestor (raw documents) and
compiler (wiki articles) both index through embed_and_upsert_chunks, so the
schema — critically the "text" field, whose absence once broke grounding
system-wide — is defined in exactly one place and cannot drift between the
two writers.
"""

import logging
from datetime import datetime, timezone
from typing import Any

from pkms.embed import embed_many
from pkms.qdrant_store import point_id, upsert_batch

logger = logging.getLogger(__name__)


def embed_and_upsert_chunks(
    collection: str,
    path: str,
    chunks: list[dict[str, Any]],
    extra_payload: dict[str, Any],
    config: dict[str, Any],
    timestamp: str | None = None,
    log_progress: bool = False,
) -> list[str]:
    """Embed every chunk (batched /api/embed) and upsert all points in one request.

    The helper writes the canonical per-chunk fields; caller-specific metadata
    (title, tags, hash, agent, …) comes in via extra_payload and may override
    the canonical ones. Returns the deterministic point ids (uuid5 of
    "path:chunk_index"), in chunk order.
    """
    if timestamp is None:
        timestamp = datetime.now(timezone.utc).isoformat()
    vectors = embed_many([ch["text"] for ch in chunks], config)
    total = len(chunks)
    points: list[tuple[str, list[float], dict[str, Any]]] = []
    for i, ch in enumerate(chunks):
        payload = {
            "path": path,
            "collection": collection,
            "section_heading": ch["section_heading"],
            # The chunk text itself — read back by compiler._chunk_texts (to
            # write grounded articles) and querier._format_hits (to answer from
            # sources). Without it every downstream consumer gets empty content
            # and falls back to headings/hallucination.
            "text": ch["text"],
            "chunk_index": i,
            "chunk_total": total,
            "timestamp": timestamp,
            **extra_payload,
        }
        points.append((point_id(path, i), vectors[i], payload))
        if log_progress:
            logger.info("Embedded chunk %d/%d%s", i + 1, total,
                        f" — {ch['section_heading']}" if ch["section_heading"] else "")
    # One request for the whole document (atomic per request on Qdrant's side)
    # instead of a round trip per chunk.
    upsert_batch(collection, points, config)
    return [pid for pid, _, _ in points]
