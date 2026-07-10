from typing import Any

import httpx

from pkms import metrics
from pkms.llm import retry_transient


class EmbedError(Exception):
    pass


# Module-level client: keep-alive across requests. A per-call httpx.post paid a
# fresh TCP setup for every embedding — hundreds per document on the old
# one-text-per-request endpoint.
_http = httpx.Client()


def _embed_batch(texts: list[str], config: dict[str, Any]) -> list[list[float]]:
    """One POST to Ollama's batch /api/embed for up to a batch of texts.

    Order-preserving (result[i] embeds texts[i]). This is the single HTTP seam
    of the module — tests mock HERE (patch pkms.embed._embed_batch), not the
    embed/embed_many wrappers. Concurrency is Ollama's job (OLLAMA_NUM_PARALLEL
    server-side); the client just batches.
    """
    host = config["ollama"]["host"]
    port = config["ollama"]["port"]
    model = config["ollama"]["models"]["embedding"]
    expected_dim = config["embedding"]["dimension"]
    timeout = config.get("ollama", {}).get("embed_timeout_seconds", 60)
    url = f"http://{host}:{port}/api/embed"

    def _attempt() -> list[list[float]]:
        resp = _http.post(url, json={"model": model, "input": texts}, timeout=timeout)
        resp.raise_for_status()
        vectors: list[list[float]] = resp.json()["embeddings"]
        if len(vectors) != len(texts):
            raise EmbedError(
                f"Expected {len(texts)} embeddings, got {len(vectors)}"
            )
        for v in vectors:
            if len(v) != expected_dim:
                raise EmbedError(
                    f"Expected embedding dimension {expected_dim}, got {len(v)}"
                )
        return vectors

    # Shared retry policy (pkms.llm.retry_transient): transient-only retries,
    # Retry-After honoured. EmbedError from validation is not transient, so it
    # propagates as-is; exhausted transient failures are wrapped for callers.
    try:
        with metrics.timer("embed_batch", n_texts=len(texts)):
            return retry_transient(_attempt, retries=3, label="embed")
    except EmbedError:
        raise
    except Exception as exc:
        raise EmbedError("Embedding failed after 3 attempts") from exc


def embed(text: str, config: dict[str, Any]) -> list[float]:
    """Embed one text using Ollama nomic-embed-text. Returns a 768-dim vector."""
    return _embed_batch([text], config)[0]


def embed_many(texts: list[str], config: dict[str, Any]) -> list[list[float]]:
    """Embed several texts via batched /api/embed requests, order-preserving.

    Batch size is `ollama.embed_batch_size` (default 32): large enough to
    amortize the HTTP round trip, small enough to keep retry granularity on
    huge documents. Any batch failing after retries raises EmbedError (same as
    a sequential loop would).
    """
    if not texts:
        return []
    batch = max(1, int(config.get("ollama", {}).get("embed_batch_size", 32)))
    out: list[list[float]] = []
    for i in range(0, len(texts), batch):
        out.extend(_embed_batch(texts[i:i + batch], config))
    return out
