"""Shared pytest fixtures."""
import pytest

import pkms.working_memory as _wm


@pytest.fixture(autouse=True)
def _reset_working_memory():
    """The working-memory session store is module-level (shared across WindowBuffer
    instances, like pkms.events._jobs). Clear it before each test so turns from one
    test's query() calls can't leak into another's."""
    _wm._reset()
    yield
    _wm._reset()
