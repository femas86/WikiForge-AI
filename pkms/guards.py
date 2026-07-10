import os
import re
from pathlib import Path
from typing import Any

import yaml

DEFAULT_PROJECT = "default"

# Project names: lowercase alphanumeric plus . _ -, must not start with a dot,
# max 64 chars. Keeps directory names safe and URL-embeddable.
_PROJECT_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{0,63}$")

# Names that clash with the pre-project vault layout (vault/raw etc.) —
# rejected to keep legacy directories from masquerading as projects.
_RESERVED_PROJECT_NAMES = {"raw", "wiki", "outputs"}

# Paths are relative to vault_root (no "vault/" prefix in the rel path computed
# by guard_write). {project} is the first path segment and must be a valid
# project name; .search-index is the single global index at the vault root.
WRITE_BOUNDARIES: dict[str, list[str]] = {
    "ingestor":    ["{project}/raw/"],
    "compiler":    ["{project}/wiki/", ".search-index"],
    "querier":     ["{project}/outputs/", ".search-index"],
    "linter":      ["{project}/outputs/"],
    "coordinator": [".search-index"],
    # The remover un-ingests a document: it deletes raw files, the derived wiki
    # article, strips dangling [[links]] in sibling articles, and prunes the index.
    "remover":     ["{project}/raw/", "{project}/wiki/", ".search-index"],
}


def normalize_project(name: str) -> str:
    """Fold user input into canonical project form: trim, lowercase, spaces → '-'.

    Validation still happens in validate_project — this only normalises the
    parts a human would not expect to matter (case, surrounding whitespace).
    """
    return (name or "").strip().lower().replace(" ", "-")


def validate_project(name: str) -> str:
    """Validate a project name; returns it unchanged or raises ValueError."""
    if not _PROJECT_NAME_RE.match(name or ""):
        raise ValueError(
            f"Invalid project name {name!r}: must match {_PROJECT_NAME_RE.pattern}"
        )
    if name in _RESERVED_PROJECT_NAMES:
        raise ValueError(f"Reserved project name {name!r}: choose another name")
    return name


def guard_write(agent: str, path: str, vault_root: str) -> None:
    abs_path = os.path.abspath(path)
    abs_vault = os.path.abspath(vault_root)
    rel = os.path.relpath(abs_path, abs_vault)
    # Normalise to forward slashes for cross-platform consistent prefix matching
    rel_fwd = rel.replace(os.sep, "/")
    allowed = WRITE_BOUNDARIES.get(agent, [])

    # Vault-root entries (e.g. ".search-index") match directly
    for boundary in allowed:
        if "{project}" not in boundary and rel_fwd.startswith(boundary.rstrip("/")):
            return

    # Project-scoped entries: first segment must be a valid project name
    parts = rel_fwd.split("/")
    if (
        len(parts) >= 2
        and _PROJECT_NAME_RE.match(parts[0])
        and parts[0] not in _RESERVED_PROJECT_NAMES
    ):
        rest = "/".join(parts[1:])
        for boundary in allowed:
            if "{project}" in boundary:
                suffix = boundary.replace("{project}/", "")
                if rest.startswith(suffix):
                    return

    raise PermissionError(
        f"[{agent}] write to '{rel_fwd}' is not in allowed boundaries: {allowed}"
    )


def list_projects(vault_dir: str | Path) -> list[str]:
    """Return existing project names — vault subdirectories with a valid name."""
    vault = Path(vault_dir)
    if not vault.exists():
        return []
    return sorted(
        d.name for d in vault.iterdir()
        if d.is_dir()
        and _PROJECT_NAME_RE.match(d.name)
        and d.name not in _RESERVED_PROJECT_NAMES
    )


# Environment overrides documented in .env.example
_ENV_OVERRIDES: list[tuple[str, str, str, type]] = [
    ("QDRANT_HOST", "qdrant", "host", str),
    ("QDRANT_PORT", "qdrant", "port", int),
    ("OLLAMA_HOST", "ollama", "host", str),
    ("OLLAMA_PORT", "ollama", "port", int),
]


def load_config(config_path: str) -> dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as fh:
        config = yaml.safe_load(fh)
    for env_var, section, key, cast in _ENV_OVERRIDES:
        value = os.environ.get(env_var)
        if value:
            config.setdefault(section, {})[key] = cast(value)
    return config
