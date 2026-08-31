"""Short-term / conversational (working) memory for the Querier (roadmap D1).

This is a SEPARATE axis from the long-term `MemoryProvider` (pkms.memory): working
memory is keyed by `session_id` and holds the recent verbatim turns of the CURRENT
conversation, so a bare follow-up ("and its complexity?") can be resolved. The
long-term provider is keyed by `user_id` and does cross-session semantic recall —
the two are orthogonal and combine freely (e.g. `memory.provider=none` +
`working_memory.strategy=window` still gets follow-up continuity).

Selected via `working_memory.strategy` in pkms.config.yaml, mirroring
`pkms.memory.get_provider`. The `window` buffer is in-process (module-level dict +
lock + inactivity eviction, modeled on pkms.events) — single-process and
non-persistent, which is fine for a single-process deployment and for dev; a
persistent (SQLite) strategy is a documented future option.
"""

from __future__ import annotations

import collections
import logging
import threading
import time
from typing import Any, Protocol, runtime_checkable

logger = logging.getLogger(__name__)

# Defaults (overridable via config `working_memory.*`).
_DEFAULT_MAX_TURNS = 6
_DEFAULT_MAX_TOKENS = 1000       # hard cap on the conversation slice injected (see querier._fit_prompt)
_DEFAULT_TTL_SECONDS = 3600      # evict a session buffer after this much inactivity
_ANSWER_CHAR_CAP = 800           # per-turn stored-answer clip so one huge answer can't bloat the buffer


@runtime_checkable
class WorkingMemory(Protocol):
    name: str
    healthy: bool

    def recall(self, session_id: str) -> list[str]:
        """Return this session's recent turns as preformatted strings, oldest→newest."""
        ...

    def append(self, session_id: str, question: str, answer: str) -> None:
        """Append a (question, answer) turn. Must be non-fatal."""
        ...


class NoneBuffer:
    """No working memory — reproduces the stateless per-turn behaviour
    (short-term memory OFF)."""

    name = "none"
    healthy = True

    def recall(self, session_id: str) -> list[str]:
        return []

    def append(self, session_id: str, question: str, answer: str) -> None:
        return None


class _SessionState:
    __slots__ = ("turns", "last_activity")

    def __init__(self, max_turns: int):
        self.turns: collections.deque[str] = collections.deque(maxlen=max_turns)
        self.last_activity = time.monotonic()


# The session store is MODULE-LEVEL (shared across WindowBuffer instances), because
# get_working_memory() builds a fresh instance per query() call — a per-instance
# store would be empty on every turn. Same pattern/rationale as pkms.events._jobs.
_SESSIONS: dict[str, _SessionState] = {}
_SESSIONS_LOCK = threading.Lock()


def _reset() -> None:
    """Clear the shared session store (tests only)."""
    with _SESSIONS_LOCK:
        _SESSIONS.clear()


class WindowBuffer:
    """Last-N verbatim turns per session, in-process. Ring buffer bounded by
    `max_turns`; sessions idle beyond `ttl` are evicted lazily on access (the
    pkms.events registry pattern). Storage is the module-level `_SESSIONS`."""

    name = "window"
    healthy = True

    def __init__(self, max_turns: int = _DEFAULT_MAX_TURNS, max_tokens: int = _DEFAULT_MAX_TOKENS,
                 ttl: int = _DEFAULT_TTL_SECONDS):
        self.max_turns = max(1, int(max_turns))
        self.max_tokens = int(max_tokens)
        self.ttl = int(ttl)

    def _evict_stale(self) -> None:
        """Drop sessions idle longer than ttl. Caller holds _SESSIONS_LOCK."""
        now = time.monotonic()
        for sid, state in list(_SESSIONS.items()):
            if now - state.last_activity > self.ttl:
                del _SESSIONS[sid]

    def recall(self, session_id: str) -> list[str]:
        if not session_id:
            return []
        try:
            with _SESSIONS_LOCK:
                self._evict_stale()
                state = _SESSIONS.get(session_id)
                return list(state.turns) if state else []
        except Exception as exc:  # non-fatal by contract
            logger.warning("Working-memory recall failed (non-fatal): %s", exc)
            return []

    def append(self, session_id: str, question: str, answer: str) -> None:
        if not session_id:
            return None
        try:
            answer = (answer or "").strip()
            if len(answer) > _ANSWER_CHAR_CAP:
                answer = answer[:_ANSWER_CHAR_CAP] + " …[clipped]"
            turn = f"Q: {(question or '').strip()}\nA: {answer}"
            with _SESSIONS_LOCK:
                self._evict_stale()
                state = _SESSIONS.get(session_id)
                if state is None:
                    state = _SessionState(self.max_turns)
                    _SESSIONS[session_id] = state
                state.turns.append(turn)
                state.last_activity = time.monotonic()
        except Exception as exc:  # non-fatal by contract
            logger.warning("Working-memory append failed (non-fatal): %s", exc)
        return None


_STRATEGIES: dict[str, type] = {
    "none": NoneBuffer,
    "window": WindowBuffer,
    # future: "summary" (summarize older turns instead of dropping them)
}


def get_working_memory(config: dict[str, Any]) -> WorkingMemory:
    """Build the configured working-memory strategy. Defaults to 'window'.

    Unknown strategy → loud warning + NoneBuffer (mirrors pkms.memory.get_provider)."""
    wm = config.get("working_memory") or {}
    name = wm.get("strategy", "window")
    cls = _STRATEGIES.get(name)
    if cls is None:
        logger.warning("Unknown working_memory strategy %r — falling back to 'none'", name)
        return NoneBuffer()
    if cls is WindowBuffer:
        return WindowBuffer(
            max_turns=wm.get("max_turns", _DEFAULT_MAX_TURNS),
            max_tokens=wm.get("max_tokens", _DEFAULT_MAX_TOKENS),
            ttl=wm.get("ttl", _DEFAULT_TTL_SECONDS),
        )
    return cls()
