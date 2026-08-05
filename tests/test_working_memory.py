"""D1 working-memory (short-term/conversational) buffer tests."""
import time

import pkms.working_memory as wm_module
from pkms.working_memory import (
    NoneBuffer,
    WindowBuffer,
    get_working_memory,
)


# ── WindowBuffer ────────────────────────────────────────────────────────────────

def test_append_then_recall_in_order():
    buf = WindowBuffer(max_turns=6)
    buf.append("s1", "what is X?", "X is a thing.")
    buf.append("s1", "and Y?", "Y is another.")
    turns = buf.recall("s1")
    assert len(turns) == 2
    assert turns[0].startswith("Q: what is X?")
    assert "A: X is a thing." in turns[0]
    assert turns[1].startswith("Q: and Y?")  # oldest→newest


def test_maxlen_drops_oldest():
    buf = WindowBuffer(max_turns=2)
    buf.append("s1", "q1", "a1")
    buf.append("s1", "q2", "a2")
    buf.append("s1", "q3", "a3")
    turns = buf.recall("s1")
    assert len(turns) == 2
    assert "q1" not in "\n".join(turns)      # oldest evicted
    assert "q2" in turns[0] and "q3" in turns[1]


def test_sessions_are_isolated():
    buf = WindowBuffer()
    buf.append("alice", "qa", "aa")
    buf.append("bob", "qb", "ab")
    assert len(buf.recall("alice")) == 1
    assert "qa" in buf.recall("alice")[0]
    assert "qb" in buf.recall("bob")[0]


def test_recall_unknown_session_is_empty():
    assert WindowBuffer().recall("never-seen") == []


def test_empty_session_id_is_noop():
    buf = WindowBuffer()
    buf.append("", "q", "a")          # no session → nothing stored
    assert buf.recall("") == []


def test_answer_is_clipped():
    buf = WindowBuffer()
    buf.append("s1", "q", "z" * 5000)
    turn = buf.recall("s1")[0]
    assert "[clipped]" in turn
    assert len(turn) < 1200           # well under the raw 5000


def test_stale_session_evicted_on_access():
    buf = WindowBuffer(ttl=3600)
    buf.append("s1", "q", "a")
    # force staleness without sleeping: push last_activity into the past
    wm_module._SESSIONS["s1"].last_activity = time.monotonic() - 4000
    assert buf.recall("s1") == []
    assert "s1" not in wm_module._SESSIONS   # actually removed, not just hidden


def test_non_fatal_on_internal_error(monkeypatch):
    buf = WindowBuffer()
    monkeypatch.setattr(wm_module, "_SESSIONS", None)  # break internals → any access raises
    assert buf.recall("s1") == []     # swallowed → []
    buf.append("s1", "q", "a")        # swallowed → no raise


# ── NoneBuffer ──────────────────────────────────────────────────────────────────

def test_none_buffer_is_noop():
    nb = NoneBuffer()
    nb.append("s1", "q", "a")
    assert nb.recall("s1") == []
    assert nb.name == "none"


# ── factory ───────────────────────────────────────────────────────────────────

def test_factory_default_is_window():
    wm = get_working_memory({})
    assert isinstance(wm, WindowBuffer)
    assert wm.name == "window"


def test_factory_explicit_none():
    wm = get_working_memory({"working_memory": {"strategy": "none"}})
    assert isinstance(wm, NoneBuffer)


def test_factory_unknown_falls_back_to_none():
    wm = get_working_memory({"working_memory": {"strategy": "bogus"}})
    assert isinstance(wm, NoneBuffer)


def test_factory_passes_config_knobs():
    wm = get_working_memory({"working_memory": {"strategy": "window", "max_turns": 3,
                                                "max_tokens": 500, "ttl": 60}})
    assert isinstance(wm, WindowBuffer)
    assert wm.max_turns == 3 and wm.max_tokens == 500 and wm.ttl == 60
