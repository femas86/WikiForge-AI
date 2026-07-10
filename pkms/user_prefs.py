"""Per-user preferences — currently the querier answer style.

Stored as plain markdown at ``~/.pkms/users/{user_id}.md``. This customizes ONLY
how the Querier formats answers for a given user; it never touches the shared,
LLM-owned wiki (that is governed per-project by vault/{project}/CLAUDE.md).
"""

from __future__ import annotations

import re
from pathlib import Path

# Mirrors the sanitization used for session ids in querier._write_output, so a
# user_id can never escape the users/ directory via path separators.
_SAFE_RE = re.compile(r"[^\w\-]")


def _users_dir() -> Path:
    return Path.home() / ".pkms" / "users"


def _user_file(user_id: str) -> Path:
    safe = _SAFE_RE.sub("_", user_id)[:60] or "default"
    return _users_dir() / f"{safe}.md"


def list_users() -> list[str]:
    """Return known user ids (those with a saved style) plus 'default', sorted.

    Filenames are already sanitized on write, so the stems are safe ids.
    """
    users = {"default"}
    d = _users_dir()
    if d.exists():
        for f in d.glob("*.md"):
            users.add(f.stem)
    return sorted(users)


def load_user_style(user_id: str) -> str:
    """Return the user's answer-style instructions, or "" if none set."""
    path = _user_file(user_id)
    if path.exists():
        return path.read_text(encoding="utf-8").strip()
    return ""


def save_user_style(user_id: str, text: str) -> Path:
    """Persist the user's answer-style instructions. Returns the file path."""
    path = _user_file(user_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.strip() + "\n", encoding="utf-8")
    return path
