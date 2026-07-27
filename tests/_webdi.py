"""Test helper for the web DI seam (roadmap B4 stage 2b).

`_override` is a context manager that installs a single FastAPI dependency
override and restores the previous state on exit — the replacement for the old
`patch.object(web_module, "_VAULT_ROOT"/"_get_config", …)` pattern. It slots into
a `with A, B:` chain exactly like the patch objects it replaces.
"""
from contextlib import contextmanager

from pkms.web import app


@contextmanager
def _override(dep, fn):
    prev = app.dependency_overrides.get(dep)
    app.dependency_overrides[dep] = fn
    try:
        yield
    finally:
        if prev is None:
            app.dependency_overrides.pop(dep, None)
        else:
            app.dependency_overrides[dep] = prev
