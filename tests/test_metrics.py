from unittest.mock import MagicMock, patch

from pkms import metrics
from pkms.metrics import collect, record, summarize, timer, usage_int


# ── collector scoping ─────────────────────────────────────────────────────────

def test_record_without_collector_is_noop():
    record("llm_call", backend="x")   # must not raise, must not leak anywhere
    with collect() as events:
        pass
    assert events == []               # nothing from outside the scope


def test_collect_captures_records_in_scope():
    with collect() as events:
        record("llm_call", backend="ollama", tokens_in=10, tokens_out=5)
        record("embed_batch", n_texts=3)
    assert len(events) == 2
    assert events[0]["kind"] == "llm_call" and events[0]["tokens_in"] == 10


def test_nested_collect_scopes_are_isolated():
    with collect() as outer:
        record("llm_call")
        with collect() as inner:
            record("embed_batch")
        record("llm_call")
    assert [e["kind"] for e in outer] == ["llm_call", "llm_call"]
    assert [e["kind"] for e in inner] == ["embed_batch"]


def test_timer_records_duration():
    with collect() as events:
        with timer("retrieval", phase="dual_search"):
            pass
    assert events[0]["kind"] == "retrieval"
    assert events[0]["phase"] == "dual_search"
    assert isinstance(events[0]["duration_ms"], float)


# ── usage coercion + summary ──────────────────────────────────────────────────

def test_usage_int_accepts_ints_rejects_junk():
    assert usage_int(42) == 42
    assert usage_int(None) is None
    assert usage_int("42") is None
    assert usage_int(True) is None        # bool is not a token count
    assert usage_int(MagicMock()) is None  # mock responses must degrade to None


def test_summarize_aggregates_llm_calls_and_durations():
    events = [
        {"kind": "llm_call", "tokens_in": 100, "tokens_out": 50, "duration_ms": 10.0},
        {"kind": "llm_call", "tokens_in": None, "tokens_out": None, "duration_ms": 5.0},
        {"kind": "retrieval", "duration_ms": 3.0},
        {"kind": "retrieval", "duration_ms": 2.0},
    ]
    s = summarize(events)
    assert s["llm_calls"] == 2            # the usage-less call still counts
    assert s["tokens_in"] == 100 and s["tokens_out"] == 50
    assert s["durations_ms"]["llm_call"] == 15.0
    assert s["durations_ms"]["retrieval"] == 5.0


# ── backend usage capture ─────────────────────────────────────────────────────

def test_call_ollama_records_native_token_counts():
    from pkms.llm import _call_ollama
    cfg = {"llm_router": {"models": {"ollama": "mistral:7b"}},
           "ollama": {"host": "localhost", "port": 11434}}
    resp = MagicMock()
    resp.json.return_value = {"response": "ok", "prompt_eval_count": 120, "eval_count": 34}
    resp.raise_for_status = MagicMock()
    with patch("pkms.llm.httpx.post", return_value=resp), collect() as events:
        out = _call_ollama("p", None, cfg)
    assert out == "ok"
    (e,) = events
    assert e["kind"] == "llm_call" and e["backend"] == "ollama"
    assert e["tokens_in"] == 120 and e["tokens_out"] == 34


def test_call_groq_records_usage():
    import os
    from pkms.llm import _call_groq
    cfg = {"llm_router": {"models": {"groq": "openai/gpt-oss-120b"}}}
    resp = MagicMock()
    resp.json.return_value = {
        "choices": [{"message": {"content": "ok"}}],
        "usage": {"prompt_tokens": 200, "completion_tokens": 80},
    }
    resp.raise_for_status = MagicMock()
    with patch.dict(os.environ, {"GROQ_API_KEY": "gk-test"}), \
         patch("pkms.llm.httpx.post", return_value=resp), collect() as events:
        _call_groq("p", None, cfg)
    (e,) = events
    assert e["backend"] == "groq"
    assert e["tokens_in"] == 200 and e["tokens_out"] == 80


def test_call_ollama_degrades_to_none_without_usage():
    # a response with no eval counts (or a test mock) must not break the call
    from pkms.llm import _call_ollama
    cfg = {"llm_router": {"models": {"ollama": "mistral:7b"}},
           "ollama": {"host": "localhost", "port": 11434}}
    resp = MagicMock()
    resp.json.return_value = {"response": "ok"}
    resp.raise_for_status = MagicMock()
    with patch("pkms.llm.httpx.post", return_value=resp), collect() as events:
        _call_ollama("p", None, cfg)
    assert events[0]["tokens_in"] is None and events[0]["tokens_out"] is None
