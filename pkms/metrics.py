"""Lightweight in-process metrics for LLM, embedding and query operations.

ContextVar-scoped collector: the owner of an operation (e.g. querier.query)
opens `collect()`; code underneath (llm backends, embed) calls `record()`
without any signature change. With no active collector, `record()` is a no-op
— instrumentation never affects a caller that doesn't ask for it.

Purpose: observability. A system that spends tokens should count them —
per-operation token cost and latency, with the native usage numbers the
backends already return (Anthropic `response.usage`, Groq `usage.*`, Ollama
`eval_count`) surfaced on the query result instead of being thrown away.
"""

import time
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Iterator

_collector: ContextVar[list | None] = ContextVar("pkms_metrics", default=None)


@contextmanager
def collect() -> Iterator[list[dict[str, Any]]]:
    """Activate a collector for this context; yields the (live) event list."""
    events: list[dict[str, Any]] = []
    token = _collector.set(events)
    try:
        yield events
    finally:
        _collector.reset(token)


def record(kind: str, **fields: Any) -> None:
    """Append an event to the active collector; no-op when none is active."""
    events = _collector.get()
    if events is not None:
        events.append({"kind": kind, **fields})


@contextmanager
def timer(kind: str, **fields: Any) -> Iterator[None]:
    """Time a block and record it as an event with duration_ms."""
    t0 = time.perf_counter()
    try:
        yield
    finally:
        record(kind, duration_ms=round((time.perf_counter() - t0) * 1000, 1), **fields)


def usage_int(value: Any) -> int | None:
    """Coerce a backend usage figure to int, or None if absent/not a number.

    Backends (and test mocks) may omit usage or return odd types; metrics must
    degrade to None, never raise.
    """
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def summarize(events: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate an event list: LLM calls/tokens plus per-kind total durations.

    Token totals are None-safe: calls whose backend reported no usage count as
    0 tokens but still increment llm_calls (so a mismatch is visible).
    """
    out: dict[str, Any] = {"llm_calls": 0, "tokens_in": 0, "tokens_out": 0,
                           "durations_ms": {}}
    for e in events:
        if e["kind"] == "llm_call":
            out["llm_calls"] += 1
            out["tokens_in"] += e.get("tokens_in") or 0
            out["tokens_out"] += e.get("tokens_out") or 0
        d = e.get("duration_ms")
        if d is not None:
            out["durations_ms"][e["kind"]] = round(
                out["durations_ms"].get(e["kind"], 0.0) + d, 1)
    return out
