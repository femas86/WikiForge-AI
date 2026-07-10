import os
from unittest.mock import patch

from pkms.memory import AMemProvider, Mem0Provider, MemPalaceProvider, NoneProvider, get_provider


def test_default_provider_is_none():
    assert isinstance(get_provider({}), NoneProvider)
    assert isinstance(get_provider({"memory": {}}), NoneProvider)


def test_get_provider_selects_by_name():
    assert isinstance(get_provider({"memory": {"provider": "none"}}), NoneProvider)
    assert isinstance(get_provider({"memory": {"provider": "mem0"}}), Mem0Provider)


def test_unknown_provider_falls_back_to_none():
    assert isinstance(get_provider({"memory": {"provider": "bogus"}}), NoneProvider)


def test_none_provider_is_inert():
    p = NoneProvider()
    assert p.recall("q", "alice", {}) == []
    # store must be a no-op and never raise
    assert p.store("q", "a", ["s"], "alice", "sess", {}, project="demo") is None


def test_mem0_provider_delegates_to_querier_helpers():
    with patch.dict(os.environ, {"MEM0_API_KEY": "m0-test"}):
        p = Mem0Provider()
    assert p.healthy
    with patch("pkms.querier._mem0_recall", return_value=["m1"]) as recall, \
         patch("pkms.querier._mem0_store") as store:
        assert p.recall("q", "alice", {"k": 1}) == ["m1"]
        p.store("q", "a", ["s"], "alice", "sess", {"k": 1}, project="demo")
    recall.assert_called_once_with("q", "alice", {"k": 1})
    store.assert_called_once_with("q", "a", ["s"], "alice", "sess", {"k": 1}, project="demo")


# ── precondition validation (healthy flag) ────────────────────────────────────

def test_mem0_without_api_key_is_unhealthy_and_noop(caplog):
    with patch.dict(os.environ, {}, clear=True):
        with caplog.at_level("ERROR", logger="pkms.memory"):
            p = Mem0Provider()
    assert not p.healthy
    assert any("MEM0_API_KEY" in r.message for r in caplog.records)   # ONE loud error
    # recall/store are fast no-ops: the querier helpers must never be reached
    with patch("pkms.querier._mem0_recall") as recall, \
         patch("pkms.querier._mem0_store") as store:
        assert p.recall("q", "alice", {}) == []
        assert p.store("q", "a", [], "alice", "s", {}) is None
    recall.assert_not_called()
    store.assert_not_called()


def test_mem0_without_sdk_is_unhealthy(caplog):
    with patch.dict(os.environ, {"MEM0_API_KEY": "m0-test"}), \
         patch("pkms.memory._module_available", return_value=False):
        with caplog.at_level("ERROR", logger="pkms.memory"):
            p = Mem0Provider()
    assert not p.healthy
    assert any("SDK" in r.message for r in caplog.records)


def test_mempalace_without_package_is_unhealthy_and_noop(caplog):
    with patch("pkms.memory._module_available", return_value=False):
        with caplog.at_level("ERROR", logger="pkms.memory"):
            p = MemPalaceProvider()
    assert not p.healthy
    assert p.recall("q", "alice", {}) == []
    assert p.store("q", "a", [], "alice", "s", {}) is None


def test_amem_healthy_when_numpy_present():
    p = AMemProvider()   # numpy is a project dependency
    assert p.healthy


def test_none_provider_always_healthy():
    assert NoneProvider().healthy
