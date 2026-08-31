import logging
import os
import re
import time
from typing import Any

import anthropic
import httpx

from pkms import metrics

logger = logging.getLogger(__name__)


class LLMError(Exception):
    pass


class LLMQuotaExhausted(LLMError):
    """A DAILY quota (RPD/TPD) is exhausted, not a per-minute (RPM/TPM) blip: the
    reset is far in the future, so retrying is futile. Raised so callers (the F2 eval)
    can ABORT the run cleanly instead of grinding a long 429 wait into every probe."""


# ── Groq rate-limit awareness (strategy A+B, see docs/f2_harness_design.md) ────────
# Groq returns per-request headers: x-ratelimit-{limit,remaining,reset}-{requests,tokens}
# — requests = RPD (daily), tokens = TPM (per-minute). We remember the last-seen TPM
# state per model to PACE proactively (avoid the 429 at all), and read retry-after /
# remaining-requests on a 429 to tell a per-minute blip (wait, capped) from a daily
# exhaustion (abort). Pacing is opt-in via llm_router.groq_pace (the eval sets it).
_GROQ_RATE: dict[str, dict[str, float]] = {}
# Two DIFFERENT thresholds (learned from live Groq Free-tier 429s):
#  - pacing may proactively sleep up to this to wait out a per-minute TPM window.
#    Groq's TPM backoff can be ~120s under sustained overage, so this must exceed it.
_PACE_WAIT_CAP_S = 180.0
#  - a 429 is treated as a DAILY (RPD/TPD) exhaustion only when requests are actually
#    gone OR the server asks to wait this long. 120s is a NORMAL per-minute backoff —
#    NOT daily — so the daily threshold sits well above it.
_DAILY_RETRY_AFTER_S = 600.0


def _parse_duration(v: str | None) -> float | None:
    """Groq reset headers look like '1h26m24s', '37m26.4s', '90ms', '2m59.56s'.
    Return seconds, or None if unparseable."""
    if not v:
        return None
    s = str(v).strip()
    try:                                  # plain seconds (retry-after is numeric)
        return float(s)
    except ValueError:
        pass
    total, matched = 0.0, False
    for value, unit in re.findall(r"(\d+(?:\.\d+)?)\s*(ms|h|m|s)", s):
        matched = True
        f = float(value)
        total += {"h": 3600.0, "m": 60.0, "s": 1.0, "ms": 0.001}[unit] * f
    return total if matched else None


def _note_groq_headers(model_id: str, headers: Any) -> None:
    """Remember this model's TPM state (remaining tokens + when the minute resets)."""
    try:
        rem = headers.get("x-ratelimit-remaining-tokens")
        if rem is None:
            return
        reset = _parse_duration(headers.get("x-ratelimit-reset-tokens")) or 0.0
        rem_req = headers.get("x-ratelimit-remaining-requests")
        _GROQ_RATE[model_id] = {
            "rem_tokens": float(rem),
            "reset_at": time.monotonic() + reset,
            "rem_requests": float(rem_req) if rem_req is not None else -1.0,
        }
    except Exception:                     # header shape drift must never break a call
        pass


def _groq_pace(model_id: str, est_tokens: int) -> None:
    """Strategy A: if the last response said the per-minute token budget can't fit this
    call, sleep until it resets (capped) BEFORE calling — turning a 429+retry into a
    quiet wait."""
    st = _GROQ_RATE.get(model_id)
    if not st or st["rem_tokens"] >= est_tokens:
        return
    wait = st["reset_at"] - time.monotonic()
    if 0 < wait <= _PACE_WAIT_CAP_S:
        logger.info("Groq %s: pacing %.1fs (TPM remaining %.0f < ~%d needed)",
                    model_id, wait, st["rem_tokens"], est_tokens)
        time.sleep(wait)


def _retry_after_seconds(exc: Exception) -> float | None:
    """Seconds to wait from a 429/503 Retry-After header, if the error carries one.

    Works for httpx.HTTPStatusError and any SDK error exposing `.response` (e.g.
    anthropic). Returns None when there's no such header (caller uses its normal
    backoff) or when it's an HTTP-date form (we don't parse dates)."""
    resp = getattr(exc, "response", None)
    if getattr(resp, "status_code", None) not in (429, 503):
        return None
    val = (getattr(resp, "headers", {}) or {}).get("retry-after")
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


def _backend_for_agent(agent: str, config: dict[str, Any]) -> str:
    agents_cfg = config["llm_router"]["agents"]
    backend = agents_cfg.get(agent)
    if isinstance(backend, dict):
        # e.g. querier: {synthesis: claude} — flatten to first value
        backend = next(iter(backend.values()))
    if backend is None:
        raise LLMError(f"No LLM backend configured for agent '{agent}'")
    return backend


def _fallback_for(backend: str, config: dict[str, Any]) -> str | None:
    return config["llm_router"].get("fallback", {}).get(backend)


def _call_claude(prompt: str, system: str | None, config: dict[str, Any],
                 temperature: float | None = None, model: str | None = None,
                 num_predict: int | None = None, response_format: str | None = None) -> str:
    # response_format is an Ollama-only hint; Claude already honours the "JSON only"
    # instruction in the prompt, so it's accepted and ignored here.
    # max_retries: the SDK honours the server's Retry-After on 429/overloaded, so a
    # generous count lets it ride out a rate-limit window instead of hard-failing
    # (the default of 2 is too low for a massive recompile). Config-tunable.
    max_retries = config["llm_router"].get("claude_max_retries", 8)
    client = anthropic.Anthropic(max_retries=max_retries)
    kwargs: dict[str, Any] = {
        "model": model or config["llm_router"]["models"]["claude"],
        "max_tokens": num_predict or 4096,
        "messages": [{"role": "user", "content": prompt}],
    }
    if system:
        kwargs["system"] = system
    if temperature is not None:
        kwargs["temperature"] = temperature
    t0 = time.perf_counter()
    response = client.messages.create(**kwargs)
    usage = getattr(response, "usage", None)
    metrics.record(
        "llm_call", backend="claude", model=kwargs["model"],
        duration_ms=round((time.perf_counter() - t0) * 1000, 1),
        tokens_in=metrics.usage_int(getattr(usage, "input_tokens", None)),
        tokens_out=metrics.usage_int(getattr(usage, "output_tokens", None)),
    )
    return response.content[0].text


def _post_json(url: str, payload: dict[str, Any], timeout: float,
               headers: dict[str, str] | None = None) -> dict[str, Any]:
    """Shared HTTP scaffolding for the OpenAI-compatible/REST backends:
    POST JSON, raise on status, return the decoded body."""
    resp = httpx.post(url, json=payload, headers=headers, timeout=timeout)
    resp.raise_for_status()
    return resp.json()


def _call_ollama(prompt: str, system: str | None, config: dict[str, Any],
                 temperature: float | None = None, model: str | None = None,
                 num_predict: int | None = None, response_format: str | None = None) -> str:
    host = config["ollama"]["host"]
    port = config["ollama"]["port"]
    url = f"http://{host}:{port}/api/generate"
    payload: dict[str, Any] = {
        "model": model or config["llm_router"]["models"]["ollama"],
        "prompt": prompt,
        "stream": False,
    }
    if system:
        payload["system"] = system
    if response_format:
        # e.g. "json" — Ollama constrains the output to valid JSON, which small
        # models (llama3.2:1b) otherwise fail to emit cleanly for metadata.
        payload["format"] = response_format
    options: dict[str, Any] = {}
    if temperature is not None:
        options["temperature"] = temperature
    if num_predict is not None:
        options["num_predict"] = num_predict
    if options:
        payload["options"] = options
    # Generation on CPU (esp. full-article compiles) can exceed the old 120s; the
    # timeout is config-tunable so slow local models don't fail spuriously.
    timeout = config.get("ollama", {}).get("timeout_seconds", 300)
    t0 = time.perf_counter()
    body = _post_json(url, payload, timeout)
    metrics.record(
        "llm_call", backend="ollama", model=payload["model"],
        duration_ms=round((time.perf_counter() - t0) * 1000, 1),
        tokens_in=metrics.usage_int(body.get("prompt_eval_count")),
        tokens_out=metrics.usage_int(body.get("eval_count")),
    )
    return body["response"]


def _call_groq(prompt: str, system: str | None, config: dict[str, Any],
               temperature: float | None = None, model: str | None = None,
               num_predict: int | None = None, response_format: str | None = None) -> str:
    """Groq's OpenAI-compatible chat-completions endpoint (httpx, no extra dep)."""
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise LLMError("GROQ_API_KEY not set — cannot call the Groq backend")
    model_id = model or config["llm_router"]["models"]["groq"]
    messages: list[dict[str, str]] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    payload: dict[str, Any] = {
        "model": model_id,
        "messages": messages,
        "max_tokens": num_predict or 4096,
    }
    if temperature is not None:
        payload["temperature"] = temperature
    if response_format == "json":
        payload["response_format"] = {"type": "json_object"}
    # Qwen3.6 is a hybrid-thinking model: left to default it prepends chain-of-thought
    # that breaks our strict output parsers (the compiler's YAML frontmatter, the
    # querier's trailing JSON block). Force non-thinking + hidden reasoning so only the
    # clean final answer comes back. Gated on the model id — other Groq models
    # (e.g. gpt-oss) don't take these params the same way.
    if "qwen3.6" in model_id:
        payload["reasoning_effort"] = "none"
        payload["reasoning_format"] = "hidden"
    elif "gpt-oss" in model_id:
        # gpt-oss is a reasoning model: its hidden reasoning eats the output budget
        # before the answer, so a small max_tokens with response_format=json_object
        # yields a 400 json_validate_failed (the JSON never gets emitted). 'medium'
        # keeps the reasoning bounded (~240 completion tokens) so a modest cap still
        # produces a complete answer, while giving the judge more grading depth than
        # 'low'. (gpt-oss rejects 'none' — only low/medium/high are valid.)
        payload["reasoning_effort"] = "medium"
    timeout = config.get("groq", {}).get("timeout_seconds", 120)
    if config.get("llm_router", {}).get("groq_pace"):
        est = (len(prompt) + len(system or "")) // 4 + min(num_predict or 512, 1024)
        _groq_pace(model_id, est)
    t0 = time.perf_counter()
    resp = httpx.post(
        "https://api.groq.com/openai/v1/chat/completions",
        json=payload, headers={"Authorization": f"Bearer {api_key}"}, timeout=timeout,
    )
    _note_groq_headers(model_id, resp.headers)   # strategy A: remember TPM state
    if resp.status_code == 429:
        # strategy B: a per-minute (TPM/RPM) blip has a short retry-after and requests
        # left → let retry_transient wait (capped) + retry. A daily (RPD/TPD) exhaustion
        # has a long retry-after and/or zero remaining requests → abort, don't grind.
        ra = _parse_duration(resp.headers.get("retry-after"))
        try:
            rem_req = float(resp.headers.get("x-ratelimit-remaining-requests"))
        except (TypeError, ValueError):
            rem_req = None
        if (rem_req is not None and rem_req <= 0) or (ra is not None and ra >= _DAILY_RETRY_AFTER_S):
            raise LLMQuotaExhausted(
                f"Groq daily quota exhausted for {model_id} "
                f"(retry-after={ra}s, remaining-requests={rem_req})"
            )
    if resp.status_code >= 400:
        # surface Groq's error body (e.g. json_validate_failed) — raise_for_status alone
        # gives an opaque HTTPStatusError with no reason
        logger.warning("Groq %s HTTP %d: %s", model_id, resp.status_code, (resp.text or "")[:300])
    resp.raise_for_status()
    body = resp.json()
    usage = body.get("usage") if isinstance(body.get("usage"), dict) else {}
    metrics.record(
        "llm_call", backend="groq", model=model_id,
        duration_ms=round((time.perf_counter() - t0) * 1000, 1),
        tokens_in=metrics.usage_int(usage.get("prompt_tokens")),
        tokens_out=metrics.usage_int(usage.get("completion_tokens")),
    )
    return body["choices"][0]["message"]["content"]


# Backend name → caller. Add a new backend here + a models.<name> in config.
_BACKENDS = {
    "claude": _call_claude,
    "ollama": _call_ollama,
    "groq":   _call_groq,
}


def _is_credit_error(exc: Exception) -> bool:
    """True for Anthropic's billing 400 ("credit balance is too low").

    This is NOT transient — retrying or falling back to a local model that either
    hallucinates or times out only wastes time and saturates the machine. Typed:
    only the SDK's BadRequestError qualifies, with the message check confined to
    that branch — an arbitrary exception whose text happens to mention "credit
    balance" (e.g. a prompt echoed into an error) must not trip a hard fail-fast.
    """
    if not isinstance(exc, anthropic.BadRequestError):
        return False
    body = exc.body if isinstance(getattr(exc, "body", None), dict) else {}
    err = body.get("error") if isinstance(body.get("error"), dict) else {}
    msg = str(err.get("message") or exc)
    return "credit balance" in msg.lower()


def _is_transient(exc: Exception) -> bool:
    """Errors worth retrying: network/timeout trouble and 408/429/5xx responses.

    Everything else — 4xx payload/auth problems, and above all programming errors
    (KeyError from a missing config key, TypeError, …) — must NOT be retried:
    burning the retry budget on them wastes LLM-timeout waits and buries the real
    cause behind a generic backend failure.
    """
    if isinstance(exc, httpx.HTTPStatusError):
        code = exc.response.status_code
        return code in (408, 429) or code >= 500
    if isinstance(exc, httpx.HTTPError):          # timeouts, connect/transport errors
        return True
    if isinstance(exc, anthropic.APIStatusError):
        code = exc.status_code
        return code in (408, 429) or code >= 500
    if isinstance(exc, anthropic.APIConnectionError):   # includes APITimeoutError
        return True
    return False


def _is_backend_error(exc: Exception) -> bool:
    """API/backend failures that justify trying the fallback backend.

    Anything else is a bug in our own code and must propagate as itself —
    routing a KeyError through the fallback masks it behind
    "Both primary and fallback failed"."""
    return isinstance(exc, (httpx.HTTPError, anthropic.APIError, LLMError))


def retry_transient(fn, retries: int = 3, base_delay: float = 1.0, label: str = "retry"):
    """THE retry policy for every third-party call (LLM backends, embeddings,
    URL fetches). Runs fn(), retrying only transient failures (_is_transient:
    network/timeout trouble, 408/429/5xx); honours a server Retry-After capped
    at 120s, else exponential backoff. Non-transient exceptions — billing 400s,
    auth errors, our own bugs — propagate immediately with their real type.
    """
    delay = base_delay
    last_exc: Exception | None = None
    for attempt in range(retries):
        try:
            return fn()
        except Exception as exc:
            if not _is_transient(exc):
                raise
            last_exc = exc
            if attempt < retries - 1:
                ra = _retry_after_seconds(exc)
                wait = min(ra, 120.0) if ra is not None else delay
                if ra is not None:
                    logger.info("%s: rate-limited — honouring Retry-After, waiting %.0fs",
                                label, wait)
                time.sleep(wait)
                delay *= 2
    raise last_exc  # type: ignore[misc]


def _call_with_retry(
    backend: str,
    prompt: str,
    system: str | None,
    config: dict[str, Any],
    retries: int = 3,
    temperature: float | None = None,
    model: str | None = None,
    num_predict: int | None = None,
    response_format: str | None = None,
) -> str:
    caller = _BACKENDS.get(backend)
    if caller is None:
        raise LLMError(f"Unknown LLM backend '{backend}' (known: {', '.join(_BACKENDS)})")
    return retry_transient(
        lambda: caller(prompt, system, config, temperature, model, num_predict, response_format),
        retries=retries,
        label=f"Backend '{backend}'",
    )


def complete(
    agent: str,
    prompt: str,
    config: dict[str, Any],
    system: str | None = None,
    temperature: float | None = None,
    model: str | None = None,
    num_predict: int | None = None,
    retries: int | None = None,
    response_format: str | None = None,
) -> str:
    """Route a completion request to the configured backend for agent.

    temperature/num_predict: passed to the backend; None keeps its default.
    model: overrides the model **for the primary backend only** (a primary-specific
    model, e.g. a small Ollama model, must not leak to the Claude fallback).
    response_format: e.g. "json" — Ollama-only output constraint; Claude ignores it
    (it honours the JSON instruction in the prompt), so it's safe on the fallback too.
    retries: attempts on the primary before falling back (fail-fast with retries=1).
    None → `llm_router.default_retries` (default 3); the F2 eval raises this for more
    patience under Groq 429s.
    Raises LLMError if both fail (or if the primary fails and `llm_router.disable_fallback`
    is set — the F2 eval sets it so a rate-limited/​timed-out cloud call never silently
    falls back to a local model and contaminates the measured answer).
    """
    lr = config.get("llm_router") or {}
    if retries is None:
        retries = int(lr.get("default_retries", 3))
    disable_fallback = bool(lr.get("disable_fallback"))
    primary = _backend_for_agent(agent, config)
    try:
        return _call_with_retry(primary, prompt, system, config, retries=retries,
                                temperature=temperature, model=model, num_predict=num_predict,
                                response_format=response_format)
    except Exception as primary_exc:
        # A daily-quota exhaustion is not fixable by a fallback or a retry — propagate
        # it as itself so the eval can abort the run cleanly (strategy B).
        if isinstance(primary_exc, LLMQuotaExhausted):
            raise
        # Billing/credit exhaustion is not something a local fallback can fix — it
        # would hallucinate or time out and saturate the machine. Fail fast and loud
        # so the caller (and the user) sees the real cause instead of silent junk.
        if _is_credit_error(primary_exc):
            raise LLMError(
                f"Backend '{primary}' unavailable: API credit balance too low — "
                "add credits, or route this agent to Ollama."
            ) from primary_exc
        # A non-backend exception is a bug in our code (missing config key, type
        # error): re-raise as itself instead of masking it behind the fallback.
        if not _is_backend_error(primary_exc):
            raise
        if disable_fallback:
            raise LLMError(
                f"Backend '{primary}' failed and fallback is disabled "
                "(llm_router.disable_fallback)"
            ) from primary_exc
        fallback = _fallback_for(primary, config)
        if fallback is None:
            raise LLMError(
                f"Primary backend '{primary}' failed and no fallback configured"
            ) from primary_exc
        try:
            # model override is primary-specific → fallback uses its own default model
            return _call_with_retry(fallback, prompt, system, config, retries=retries,
                                    temperature=temperature, model=None, num_predict=num_predict,
                                    response_format=response_format)
        except Exception as fallback_exc:
            if not _is_backend_error(fallback_exc):
                raise
            raise LLMError(
                f"Both primary '{primary}' and fallback '{fallback}' failed"
            ) from fallback_exc
