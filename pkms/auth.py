"""B4 multi-user: identity resolution + per-project role model.

Identity is delegated to a reverse-proxy / SSO in front of the app, which sets a
trusted header (e.g. X-Auth-User) with the authenticated user id. When no proxy is
present (dev / single-user), a configurable default user is used. This module is the
single place that resolves "who is the caller"; authorization roles live per project
in the `project_members` table (see pkms.db).

Deploy-agnostic by design: only the *source* of the header (which proxy/IdP, or none)
depends on the not-yet-decided deployment — the app just reads it.
"""

from typing import Any, Mapping


class AccessDenied(Exception):
    """Raised when the caller lacks the required role on a project."""


# Roles in ascending privilege. A higher role subsumes the lower ones.
ROLES = ("viewer", "editor", "owner")
_RANK = {role: i for i, role in enumerate(ROLES)}


def role_allows(role: str | None, required: str) -> bool:
    """True if `role` is at least `required` (owner ≥ editor ≥ viewer).

    A non-member (role None) or an unknown role is never allowed.
    """
    if role is None or role not in _RANK:
        return False
    return _RANK[role] >= _RANK[required]


def current_user(headers: Mapping[str, str] | None, config: dict[str, Any]) -> str:
    """Resolve the calling user id.

    Reads the trusted auth header (config `auth.user_header`, default 'X-Auth-User')
    if present; otherwise falls back to the configured default (config
    `auth.default_user`, default 'local'). Header lookup is case-insensitive for
    plain dicts too (falls back to a manual scan). CLI callers pass headers=None.
    """
    auth = config.get("auth") or {}
    header_name = auth.get("user_header", "X-Auth-User")
    default_user = auth.get("default_user", "local")

    value = None
    if headers is not None:
        value = headers.get(header_name)
        if value is None:  # case-insensitive fallback for plain dicts
            low = header_name.lower()
            for k, v in headers.items():
                if k.lower() == low:
                    value = v
                    break

    resolved = (value or "").strip() or default_user
    return resolved
