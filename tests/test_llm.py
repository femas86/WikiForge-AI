import os
from unittest.mock import MagicMock, patch

import anthropic
import httpx
import pytest

from pkms.llm import complete, LLMError, _backend_for_agent, _fallback_for, _call_claude


# ── realistic exception builders ─────────────────────────────────────────────
# Retry/fallback are gated on exception TYPE (transient vs permanent vs bug),
# so tests must raise what the real backends raise, not bare RuntimeError.

def _conn_error(msg="connection refused"):
    return httpx.ConnectError(msg)


def _anthropic_conn_error():
    req = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    return anthropic.APIConnectionError(request=req)


def _http_status_error(code, headers=None, url="https://api.example.com/v1"):
    req = httpx.Request("POST", url)
    resp = httpx.Response(code, request=req, headers=headers or {})
    return httpx.HTTPStatusError(f"HTTP {code}", request=req, response=resp)


def _anthropic_credit_error():
    msg = "Your credit balance is too low to access the Anthropic API."
    req = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    resp = httpx.Response(400, request=req)
    return anthropic.BadRequestError(
        msg, response=resp,
        body={"error": {"type": "invalid_request_error", "message": msg}},
    )


CONFIG = {
    "llm_router": {
        "agents": {
            "compiler": "claude",
            "ingestor": "ollama",
            "querier":  {"synthesis": "claude"},
        },
        "fallback": {"claude": "ollama"},
        "models": {
            "claude": "claude-sonnet-4-6",
            "ollama": "mistral:7b",
        },
    },
    "ollama": {"host": "localhost", "port": 11434},
}

CONFIG_NO_FALLBACK = {
    "llm_router": {
        "agents": {"compiler": "claude"},
        "fallback": {},
        "models": {"claude": "claude-sonnet-4-6", "ollama": "mistral:7b"},
    },
    "ollama": {"host": "localhost", "port": 11434},
}


# ── routing helpers ───────────────────────────────────────────────────────────

def test_backend_for_agent_direct():
    assert _backend_for_agent("compiler", CONFIG) == "claude"


def test_backend_for_agent_dict_value():
    # querier: {synthesis: claude} should resolve to "claude"
    assert _backend_for_agent("querier", CONFIG) == "claude"


def test_backend_for_agent_ollama():
    assert _backend_for_agent("ingestor", CONFIG) == "ollama"


def test_backend_for_unknown_agent_raises():
    with pytest.raises(LLMError, match="No LLM backend"):
        _backend_for_agent("unknown", CONFIG)


def test_fallback_for_claude():
    assert _fallback_for("claude", CONFIG) == "ollama"


def test_fallback_for_ollama_is_none():
    assert _fallback_for("ollama", CONFIG) is None


# ── complete — Claude happy path ──────────────────────────────────────────────

def _mock_anthropic_response(text: str):
    msg = MagicMock()
    msg.content = [MagicMock(text=text)]
    return msg


def test_complete_claude_success():
    with patch("pkms.llm.anthropic.Anthropic") as MockClient:
        MockClient.return_value.messages.create.return_value = (
            _mock_anthropic_response("hello from claude")
        )
        result = complete("compiler", "say hello", CONFIG)
    assert result == "hello from claude"


def test_complete_claude_passes_system_prompt():
    with patch("pkms.llm.anthropic.Anthropic") as MockClient:
        create = MockClient.return_value.messages.create
        create.return_value = _mock_anthropic_response("ok")
        complete("compiler", "prompt", CONFIG, system="be concise")
    _, kwargs = create.call_args
    assert kwargs.get("system") == "be concise" or create.call_args[1].get("system") == "be concise"


# ── complete — Ollama happy path ──────────────────────────────────────────────

def test_complete_ollama_success():
    with patch("pkms.llm.httpx.post") as mock_post:
        mock_post.return_value.json.return_value = {"response": "hello from ollama"}
        mock_post.return_value.raise_for_status = MagicMock()
        result = complete("ingestor", "say hello", CONFIG)
    assert result == "hello from ollama"


# ── fallback behaviour ────────────────────────────────────────────────────────

def test_falls_back_to_ollama_when_claude_fails():
    with patch("pkms.llm.anthropic.Anthropic") as MockClient, \
         patch("pkms.llm.httpx.post") as mock_post:
        MockClient.return_value.messages.create.side_effect = _anthropic_conn_error()
        mock_post.return_value.json.return_value = {"response": "fallback answer"}
        mock_post.return_value.raise_for_status = MagicMock()
        result = complete("compiler", "prompt", CONFIG)
    assert result == "fallback answer"


CONFIG_SYMMETRIC = {
    "llm_router": {
        "agents": {"ingestor": "ollama"},
        "fallback": {"claude": "ollama", "ollama": "claude"},
        "models": {"claude": "claude-sonnet-4-6", "ollama": "mistral:7b"},
    },
    "ollama": {"host": "localhost", "port": 11434},
}


def test_fallback_for_ollama_when_configured():
    assert _fallback_for("ollama", CONFIG_SYMMETRIC) == "claude"


def test_falls_back_to_claude_when_ollama_fails():
    # Ollama backend errors (cold-start/overload) → router falls back to Claude.
    with patch("pkms.llm.httpx.post") as mock_post, \
         patch("pkms.llm.anthropic.Anthropic") as MockClient:
        mock_post.side_effect = _conn_error("ollama down")
        MockClient.return_value.messages.create.return_value = (
            _mock_anthropic_response("claude saved the ingest")
        )
        result = complete("ingestor", "extract metadata", CONFIG_SYMMETRIC)
    assert result == "claude saved the ingest"


def test_raises_llm_error_when_both_fail():
    with patch("pkms.llm.anthropic.Anthropic") as MockClient, \
         patch("pkms.llm.httpx.post") as mock_post:
        MockClient.return_value.messages.create.side_effect = _anthropic_conn_error()
        mock_post.side_effect = _conn_error("ollama down")
        with pytest.raises(LLMError, match="Both primary"):
            complete("compiler", "prompt", CONFIG)


def test_raises_llm_error_no_fallback_configured():
    with patch("pkms.llm.anthropic.Anthropic") as MockClient:
        MockClient.return_value.messages.create.side_effect = _anthropic_conn_error()
        with pytest.raises(LLMError, match="no fallback configured"):
            complete("compiler", "prompt", CONFIG_NO_FALLBACK)


# ── retry behaviour ───────────────────────────────────────────────────────────

def test_retries_three_times_before_fallback():
    call_count = 0

    def failing_create(**kwargs):
        nonlocal call_count
        call_count += 1
        raise _anthropic_conn_error()   # transient → retried

    with patch("pkms.llm.anthropic.Anthropic") as MockClient, \
         patch("pkms.llm.httpx.post") as mock_post, \
         patch("pkms.llm.time.sleep"):  # skip actual sleeps
        MockClient.return_value.messages.create.side_effect = failing_create
        mock_post.return_value.json.return_value = {"response": "ok"}
        mock_post.return_value.raise_for_status = MagicMock()
        result = complete("compiler", "prompt", CONFIG)

    assert call_count == 3   # 3 attempts on primary before fallback
    assert result == "ok"


def test_complete_passes_temperature_to_ollama():
    with patch("pkms.llm.httpx.post") as mock_post:
        mock_post.return_value.json.return_value = {"response": "ok"}
        mock_post.return_value.raise_for_status = MagicMock()
        complete("ingestor", "p", CONFIG, temperature=0.1)
    _, kwargs = mock_post.call_args
    assert kwargs["json"]["options"]["temperature"] == 0.1


def test_complete_omits_temperature_when_none():
    with patch("pkms.llm.httpx.post") as mock_post:
        mock_post.return_value.json.return_value = {"response": "ok"}
        mock_post.return_value.raise_for_status = MagicMock()
        complete("ingestor", "p", CONFIG)  # no temperature
    _, kwargs = mock_post.call_args
    assert "options" not in kwargs["json"]


def test_model_override_applies_to_ollama_primary():
    with patch("pkms.llm.httpx.post") as mock_post:
        mock_post.return_value.json.return_value = {"response": "ok"}
        mock_post.return_value.raise_for_status = MagicMock()
        complete("ingestor", "p", CONFIG, model="llama3.2:1b", num_predict=256)
    body = mock_post.call_args.kwargs["json"]
    assert body["model"] == "llama3.2:1b"
    assert body["options"]["num_predict"] == 256


def test_model_override_does_not_leak_to_claude_fallback():
    # ollama primary fails → fallback to claude; the ollama-specific model must NOT reach claude
    with patch("pkms.llm.httpx.post", side_effect=_conn_error("ollama down")), \
         patch("pkms.llm.anthropic.Anthropic") as MockClient:
        create = MockClient.return_value.messages.create
        create.return_value = _mock_anthropic_response("claude ok")
        out = complete("ingestor", "p", CONFIG_SYMMETRIC, model="llama3.2:1b", retries=1)
    assert out == "claude ok"
    assert create.call_args.kwargs["model"] == "claude-sonnet-4-6"   # claude default, not the override


def test_retries_one_makes_single_primary_attempt():
    calls = {"n": 0}
    def boom(*a, **k):
        calls["n"] += 1
        raise _conn_error("x")
    with patch("pkms.llm.httpx.post", side_effect=boom), \
         patch("pkms.llm.anthropic.Anthropic") as MockClient, \
         patch("pkms.llm.time.sleep"):
        MockClient.return_value.messages.create.return_value = _mock_anthropic_response("ok")
        complete("ingestor", "p", CONFIG_SYMMETRIC, retries=1)
    assert calls["n"] == 1   # one ollama attempt (fail-fast), then claude fallback


def test_call_claude_passes_configured_max_retries():
    # SDK max_retries (honours Retry-After on 429) is config-driven, default 8
    cfg = {"llm_router": {"models": {"claude": "claude-sonnet-4-6"}, "claude_max_retries": 7}}
    with patch("pkms.llm.anthropic.Anthropic") as MockClient:
        MockClient.return_value.messages.create.return_value = _mock_anthropic_response("hi")
        out = _call_claude("prompt", None, cfg)
    assert out == "hi"
    MockClient.assert_called_once_with(max_retries=7)


def test_call_claude_default_max_retries_when_unset():
    cfg = {"llm_router": {"models": {"claude": "claude-sonnet-4-6"}}}
    with patch("pkms.llm.anthropic.Anthropic") as MockClient:
        MockClient.return_value.messages.create.return_value = _mock_anthropic_response("hi")
        _call_claude("prompt", None, cfg)
    MockClient.assert_called_once_with(max_retries=8)


# ── ollama timeout + credit-error fail-fast ───────────────────────────────────

def test_call_ollama_uses_configured_timeout():
    cfg = {**CONFIG, "ollama": {**CONFIG["ollama"], "timeout_seconds": 250}}
    with patch("pkms.llm.httpx.post") as mock_post:
        mock_post.return_value.json.return_value = {"response": "ok"}
        mock_post.return_value.raise_for_status = MagicMock()
        complete("ingestor", "p", cfg)
    assert mock_post.call_args.kwargs["timeout"] == 250


def test_is_credit_error_detection():
    from pkms.llm import _is_credit_error
    assert _is_credit_error(_anthropic_credit_error())
    # typed gate: the phrase alone in an arbitrary exception must NOT trip it
    assert not _is_credit_error(RuntimeError("Your credit balance is too low to access the API"))
    # a BadRequestError about something else is not a credit error
    req = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    other_400 = anthropic.BadRequestError(
        "max_tokens too large", response=httpx.Response(400, request=req),
        body={"error": {"type": "invalid_request_error", "message": "max_tokens too large"}})
    assert not _is_credit_error(other_400)


def test_credit_error_fails_fast_no_retry_no_fallback():
    calls = {"claude": 0}

    def claude_credit_fail(**kwargs):
        calls["claude"] += 1
        raise _anthropic_credit_error()

    with patch("pkms.llm.anthropic.Anthropic") as MockClient, \
         patch("pkms.llm.httpx.post") as mock_post, \
         patch("pkms.llm.time.sleep"):
        MockClient.return_value.messages.create.side_effect = claude_credit_fail
        mock_post.return_value.json.return_value = {"response": "junk fallback"}
        mock_post.return_value.raise_for_status = MagicMock()
        with pytest.raises(LLMError, match="credit balance too low"):
            complete("compiler", "prompt", CONFIG)

    assert calls["claude"] == 1        # single attempt — no 3x retry
    mock_post.assert_not_called()      # no Ollama fallback on a billing error


# ── Groq backend ──────────────────────────────────────────────────────────────

def _groq_resp(text):
    r = MagicMock()
    r.status_code = 200
    r.headers = {}
    r.json.return_value = {"choices": [{"message": {"content": text}}]}
    r.raise_for_status = MagicMock()
    return r


def test_call_groq_forces_non_thinking_for_qwen36():
    from pkms.llm import _call_groq
    cfg = {"llm_router": {"models": {"groq": "qwen/qwen3.6-27b"}}}
    with patch.dict(os.environ, {"GROQ_API_KEY": "gk-test"}), \
         patch("pkms.llm.httpx.post", return_value=_groq_resp("ok")) as mp:
        out = _call_groq("prompt", None, cfg)
    body = mp.call_args.kwargs["json"]
    assert out == "ok"
    assert body["model"] == "qwen/qwen3.6-27b"
    assert body["reasoning_effort"] == "none"      # non-thinking forced for qwen3.6
    assert body["reasoning_format"] == "hidden"


def test_call_groq_omits_reasoning_for_other_models():
    from pkms.llm import _call_groq
    cfg = {"llm_router": {"models": {"groq": "llama-3.3-70b-versatile"}}}
    with patch.dict(os.environ, {"GROQ_API_KEY": "gk-test"}), \
         patch("pkms.llm.httpx.post", return_value=_groq_resp("ok")) as mp:
        _call_groq("prompt", None, cfg)
    body = mp.call_args.kwargs["json"]
    # reasoning params are added only for qwen3.6 (none/hidden) and gpt-oss (medium);
    # a plain model gets neither
    assert "reasoning_effort" not in body and "reasoning_format" not in body
    assert "reasoning_format" not in body


def test_call_groq_requires_api_key():
    from pkms.llm import _call_groq, LLMError
    cfg = {"llm_router": {"models": {"groq": "qwen/qwen3.6-27b"}}}
    with patch.dict(os.environ, {}, clear=True), \
         pytest.raises(LLMError, match="GROQ_API_KEY"):
        _call_groq("prompt", None, cfg)


GROQ_CFG = {
    "llm_router": {
        "agents": {"compiler": "groq"},
        "fallback": {"groq": "ollama"},
        "models": {"groq": "qwen/qwen3.6-27b", "ollama": "mistral:7b", "claude": "claude-sonnet-4-6"},
    },
    "ollama": {"host": "localhost", "port": 11434},
}


def test_complete_routes_to_groq_with_auth_header():
    with patch.dict(os.environ, {"GROQ_API_KEY": "gk-test"}), \
         patch("pkms.llm.httpx.post", return_value=_groq_resp("groq answer")) as mp:
        out = complete("compiler", "prompt", GROQ_CFG)
    assert out == "groq answer"
    assert "api.groq.com" in mp.call_args.args[0]
    assert mp.call_args.kwargs["headers"]["Authorization"] == "Bearer gk-test"


def test_complete_groq_falls_back_to_ollama():
    def router(url, **kw):
        if "api.groq.com" in url:
            raise _http_status_error(500, url=url)
        r = MagicMock()
        r.json.return_value = {"response": "ollama fallback"}
        r.raise_for_status = MagicMock()
        return r
    with patch.dict(os.environ, {"GROQ_API_KEY": "gk-test"}), \
         patch("pkms.llm.httpx.post", side_effect=router), \
         patch("pkms.llm.time.sleep"):
        out = complete("compiler", "prompt", GROQ_CFG)
    assert out == "ollama fallback"


from pkms.llm import LLMQuotaExhausted, _call_groq, _parse_duration

_GROQ_MODEL_CFG = {"llm_router": {"models": {"groq": "qwen/qwen3.6-27b"}}, "groq": {}}


def _groq_response(status, headers, body=None):
    r = MagicMock()
    r.status_code = status
    r.headers = headers
    r.json.return_value = body or {"choices": [{"message": {"content": "ok"}}], "usage": {}}
    if status >= 400:
        r.raise_for_status.side_effect = _http_status_error(status)
    else:
        r.raise_for_status.return_value = None
    return r


def test_parse_duration_forms():
    assert abs(_parse_duration("1h26m24s") - (3600 + 26 * 60 + 24)) < 1e-6
    assert abs(_parse_duration("37m26.4s") - (37 * 60 + 26.4)) < 1e-6
    assert abs(_parse_duration("90ms") - 0.09) < 1e-9
    assert _parse_duration("7") == 7.0          # plain seconds (retry-after)
    assert _parse_duration("junk") is None and _parse_duration(None) is None


def test_call_groq_daily_quota_raises_exhausted():
    # long retry-after + zero remaining requests = daily (RPD/TPD) exhaustion → abort
    resp = _groq_response(429, {"retry-after": "3600", "x-ratelimit-remaining-requests": "0",
                                "x-ratelimit-remaining-tokens": "0", "x-ratelimit-reset-tokens": "1s"})
    with patch.dict(os.environ, {"GROQ_API_KEY": "gk"}), \
         patch("pkms.llm.httpx.post", return_value=resp):
        with pytest.raises(LLMQuotaExhausted):
            _call_groq("p", None, _GROQ_MODEL_CFG)


def test_call_groq_minute_blip_stays_transient():
    # short retry-after + requests remaining = per-minute TPM blip → normal 429 (retryable)
    resp = _groq_response(429, {"retry-after": "5", "x-ratelimit-remaining-requests": "900",
                                "x-ratelimit-remaining-tokens": "0", "x-ratelimit-reset-tokens": "3s"})
    with patch.dict(os.environ, {"GROQ_API_KEY": "gk"}), \
         patch("pkms.llm.httpx.post", return_value=resp):
        with pytest.raises(httpx.HTTPStatusError):
            _call_groq("p", None, _GROQ_MODEL_CFG)


def test_call_groq_gpt_oss_sets_reasoning_effort_medium():
    # gpt-oss reasoning eats the output budget → we bound it so a small cap still emits
    # valid JSON (avoids 400 json_validate_failed)
    captured = {}

    def router(url, json=None, **kw):
        captured.update(json or {})
        return _groq_response(200, {}, {"choices": [{"message": {"content": "ok"}}], "usage": {}})

    with patch.dict(os.environ, {"GROQ_API_KEY": "gk"}), \
         patch("pkms.llm.httpx.post", side_effect=router):
        _call_groq("p", None, {"llm_router": {"models": {"groq": "openai/gpt-oss-120b"}}, "groq": {}})
    assert captured.get("reasoning_effort") == "medium"


def test_call_groq_120s_tpm_backoff_is_not_daily():
    # 120s retry-after is Groq's per-minute TPM backoff when requests are still available —
    # it must stay a retryable 429, NOT be misread as a daily-quota abort (regression).
    resp = _groq_response(429, {"retry-after": "120", "x-ratelimit-remaining-requests": "870",
                                "x-ratelimit-remaining-tokens": "0", "x-ratelimit-reset-tokens": "2m"})
    with patch.dict(os.environ, {"GROQ_API_KEY": "gk"}), \
         patch("pkms.llm.httpx.post", return_value=resp):
        with pytest.raises(httpx.HTTPStatusError):
            _call_groq("p", None, _GROQ_MODEL_CFG)


def test_complete_propagates_quota_exhausted_even_with_fallback():
    # a daily exhaustion must NOT be masked by the fallback path — it propagates as itself
    resp = _groq_response(429, {"retry-after": "7200", "x-ratelimit-remaining-requests": "0"})
    with patch.dict(os.environ, {"GROQ_API_KEY": "gk"}), \
         patch("pkms.llm.httpx.post", return_value=resp), patch("pkms.llm.time.sleep"):
        with pytest.raises(LLMQuotaExhausted):
            complete("compiler", "p", GROQ_CFG)


def test_complete_disable_fallback_raises_instead_of_falling_back():
    """F2 eval sets llm_router.disable_fallback so a failed cloud call never silently
    runs on the local model — it raises LLMError (the harness records an 'error' probe)."""
    import copy as _copy
    cfg = _copy.deepcopy(GROQ_CFG)
    cfg["llm_router"]["disable_fallback"] = True

    def router(url, **kw):
        if "api.groq.com" in url:
            raise _http_status_error(500, url=url)
        r = MagicMock()
        r.json.return_value = {"response": "ollama fallback"}
        r.raise_for_status = MagicMock()
        return r
    with patch.dict(os.environ, {"GROQ_API_KEY": "gk-test"}), \
         patch("pkms.llm.httpx.post", side_effect=router), \
         patch("pkms.llm.time.sleep"):
        with pytest.raises(LLMError, match="fallback is disabled"):
            complete("compiler", "prompt", cfg)


# ── Retry-After honouring (429/503 rate limits) ───────────────────────────────

def test_retry_after_seconds_extracts_header():
    from pkms.llm import _retry_after_seconds
    class R429:
        status_code = 429
        headers = {"retry-after": "12"}
    e = Exception(); e.response = R429()
    assert _retry_after_seconds(e) == 12.0

    class R500:
        status_code = 500
        headers = {"retry-after": "5"}
    e2 = Exception(); e2.response = R500()
    assert _retry_after_seconds(e2) is None       # only 429/503
    assert _retry_after_seconds(Exception()) is None   # no .response


def test_call_with_retry_honours_retry_after():
    from pkms.llm import _call_with_retry
    calls = {"n": 0}
    def caller(prompt, system, config, temperature, model, num_predict, response_format):
        calls["n"] += 1
        if calls["n"] == 1:
            raise _http_status_error(429, headers={"retry-after": "30"})
        return "ok"
    with patch.dict("pkms.llm._BACKENDS", {"groq": caller}), \
         patch("pkms.llm.time.sleep") as ms:
        out = _call_with_retry("groq", "p", None, {}, retries=3)
    assert out == "ok"
    ms.assert_called_once_with(30.0)   # waited the header value, not the 1.0 exponential


# ── typed error discrimination ────────────────────────────────────────────────

def test_config_bug_propagates_as_itself_no_retry_no_fallback():
    # A KeyError from a missing config key is OUR bug: it must surface unchanged
    # after ONE attempt — not be retried 3x, not be masked by the fallback.
    calls = {"n": 0}
    def buggy(prompt, system, config, temperature, model, num_predict, response_format):
        calls["n"] += 1
        raise KeyError("chunking")
    fallback_called = {"n": 0}
    def fb(*a, **k):
        fallback_called["n"] += 1
        return "should never run"
    cfg = {"llm_router": {"agents": {"compiler": "groq"},
                          "fallback": {"groq": "ollama"}, "models": {}}}
    with patch.dict("pkms.llm._BACKENDS", {"groq": buggy, "ollama": fb}), \
         patch("pkms.llm.time.sleep"):
        with pytest.raises(KeyError, match="chunking"):
            complete("compiler", "p", cfg)
    assert calls["n"] == 1
    assert fallback_called["n"] == 0


def test_permanent_4xx_not_retried_but_falls_back():
    # A 400/413 is permanent for THIS backend (no retry) but another backend may
    # not share the limit (e.g. Groq 413 payload → Ollama has no such cap).
    calls = {"groq": 0, "ollama": 0}
    def groq_413(*a, **k):
        calls["groq"] += 1
        raise _http_status_error(413)
    def ollama_ok(*a, **k):
        calls["ollama"] += 1
        return "ollama took it"
    cfg = {"llm_router": {"agents": {"compiler": "groq"},
                          "fallback": {"groq": "ollama"}, "models": {}}}
    with patch.dict("pkms.llm._BACKENDS", {"groq": groq_413, "ollama": ollama_ok}), \
         patch("pkms.llm.time.sleep"):
        out = complete("compiler", "p", cfg)
    assert out == "ollama took it"
    assert calls["groq"] == 1     # single attempt — 4xx is not transient
    assert calls["ollama"] == 1


def test_transient_5xx_is_retried():
    from pkms.llm import _call_with_retry
    calls = {"n": 0}
    def flaky(prompt, system, config, temperature, model, num_predict, response_format):
        calls["n"] += 1
        if calls["n"] < 3:
            raise _http_status_error(503)
        return "recovered"
    with patch.dict("pkms.llm._BACKENDS", {"groq": flaky}), \
         patch("pkms.llm.time.sleep"):
        assert _call_with_retry("groq", "p", None, {}, retries=3) == "recovered"
    assert calls["n"] == 3
