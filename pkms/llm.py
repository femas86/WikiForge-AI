import logging
import os
import time
from typing import Any

import anthropic
import httpx

from pkms import metrics

logger = logging.getLogger(__name__)


class LLMError(Exception):
    pass


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
    timeout = config.get("groq", {}).get("timeout_seconds", 120)
    t0 = time.perf_counter()
    body = _post_json(
        "https://api.groq.com/openai/v1/chat/completions",
        payload,
        timeout,
        headers={"Authorization": f"Bearer {api_key}"},
    )
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
    retries: int = 3,
    response_format: str | None = None,
) -> str:
    """Route a completion request to the configured backend for agent.

    temperature/num_predict: passed to the backend; None keeps its default.
    model: overrides the model **for the primary backend only** (a primary-specific
    model, e.g. a small Ollama model, must not leak to the Claude fallback).
    response_format: e.g. "json" — Ollama-only output constraint; Claude ignores it
    (it honours the JSON instruction in the prompt), so it's safe on the fallback too.
    retries: attempts on the primary before falling back (fail-fast with retries=1).
    Raises LLMError if both fail.
    """
    primary = _backend_for_agent(agent, config)
    try:
        return _call_with_retry(primary, prompt, system, config, retries=retries,
                                temperature=temperature, model=model, num_predict=num_predict,
                                response_format=response_format)
    except Exception as primary_exc:
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
