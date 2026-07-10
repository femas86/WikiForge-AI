from unittest.mock import MagicMock, patch

import httpx
import pytest

from pkms.embed import EmbedError, embed, embed_many


CONFIG = {
    "ollama": {
        "host": "localhost",
        "port": 11434,
        "models": {"embedding": "nomic-embed-text"},
    },
    "embedding": {"dimension": 768},
}


def _mock_response(vectors: list[list[float]]):
    resp = MagicMock()
    resp.json.return_value = {"embeddings": vectors}
    resp.raise_for_status = MagicMock()
    return resp


# ── single-text wrapper ───────────────────────────────────────────────────────

def test_embed_returns_vector():
    vector = [0.1] * 768
    with patch("pkms.embed._http.post", return_value=_mock_response([vector])):
        result = embed("hello world", CONFIG)
    assert result == vector
    assert len(result) == 768


def test_embed_posts_batch_payload_to_api_embed():
    vector = [0.0] * 768
    with patch("pkms.embed._http.post", return_value=_mock_response([vector])) as mock_post:
        embed("test text", CONFIG)
    args, kwargs = mock_post.call_args
    assert args[0].endswith("/api/embed")          # batch endpoint, not legacy /api/embeddings
    assert kwargs["json"]["model"] == "nomic-embed-text"
    assert kwargs["json"]["input"] == ["test text"]


def test_embed_uses_configured_timeout():
    with patch("pkms.embed._http.post", return_value=_mock_response([[0.1] * 768])) as mock_post:
        embed("x", {**CONFIG, "ollama": {**CONFIG["ollama"], "embed_timeout_seconds": 99}})
    assert mock_post.call_args.kwargs["timeout"] == 99


# ── validation ────────────────────────────────────────────────────────────────

def test_embed_wrong_dimension_raises():
    bad = [[0.1] * 512]  # wrong dim
    with patch("pkms.embed._http.post", return_value=_mock_response(bad)):
        with pytest.raises(EmbedError, match="dimension"):
            embed("text", CONFIG)


def test_embed_count_mismatch_raises():
    # the server must return one vector per input text
    with patch("pkms.embed._http.post", return_value=_mock_response([[0.1] * 768])):
        with pytest.raises(EmbedError, match="Expected 2 embeddings"):
            embed_many(["a", "b"], CONFIG)


def test_embed_http_error_raises_embed_error():
    resp = MagicMock()
    resp.raise_for_status.side_effect = httpx.HTTPStatusError(
        "500", request=MagicMock(), response=MagicMock()
    )
    with patch("pkms.embed._http.post", return_value=resp), \
         patch("pkms.llm.time.sleep"):
        with pytest.raises(EmbedError):
            embed("text", CONFIG)


# ── retry behaviour ───────────────────────────────────────────────────────────

def test_embed_retries_on_failure_then_succeeds():
    vector = [0.5] * 768
    call_count = 0

    def flaky_post(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count < 3:
            raise httpx.TimeoutException("timeout")
        return _mock_response([vector])

    with patch("pkms.embed._http.post", side_effect=flaky_post), \
         patch("pkms.llm.time.sleep"):
        result = embed("text", CONFIG)

    assert call_count == 3
    assert result == vector


def test_embed_raises_after_three_failures():
    with patch("pkms.embed._http.post", side_effect=httpx.TimeoutException("timeout")), \
         patch("pkms.llm.time.sleep"):
        with pytest.raises(EmbedError, match="3 attempts"):
            embed("text", CONFIG)


def test_embed_honours_retry_after_header():
    req = httpx.Request("POST", "http://localhost:11434/api/embed")
    resp429 = httpx.Response(429, request=req, headers={"retry-after": "17"})
    calls = {"n": 0}

    def limited_post(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            raise httpx.HTTPStatusError("429", request=req, response=resp429)
        return _mock_response([[0.2] * 768])

    with patch("pkms.embed._http.post", side_effect=limited_post), \
         patch("pkms.llm.time.sleep") as ms:
        embed("text", CONFIG)
    ms.assert_called_once_with(17.0)   # header value, not the 1.0 exponential


# ── embed_many batching ───────────────────────────────────────────────────────

def test_embed_many_empty_returns_empty():
    assert embed_many([], CONFIG) == []


def test_embed_many_single_request_within_batch_size():
    vecs = [[float(i)] * 768 for i in range(3)]
    with patch("pkms.embed._http.post", return_value=_mock_response(vecs)) as mock_post:
        result = embed_many(["a", "b", "c"], CONFIG)
    assert result == vecs                              # order preserved
    assert mock_post.call_count == 1                   # one batch → one request
    assert mock_post.call_args.kwargs["json"]["input"] == ["a", "b", "c"]


def test_embed_many_splits_into_batches():
    cfg = {**CONFIG, "ollama": {**CONFIG["ollama"], "embed_batch_size": 2}}
    seen_inputs = []

    def batch_post(url, json, timeout):
        seen_inputs.append(list(json["input"]))
        return _mock_response([[0.1] * 768] * len(json["input"]))

    with patch("pkms.embed._http.post", side_effect=batch_post):
        result = embed_many(["a", "b", "c", "d", "e"], cfg)

    assert len(result) == 5
    assert seen_inputs == [["a", "b"], ["c", "d"], ["e"]]   # 2+2+1


def test_embed_many_batch_failure_propagates():
    with patch("pkms.embed._http.post", side_effect=httpx.ConnectError("down")), \
         patch("pkms.llm.time.sleep"):
        with pytest.raises(EmbedError):
            embed_many(["a", "b"], CONFIG)
