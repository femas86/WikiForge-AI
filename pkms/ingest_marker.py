"""Cross-process ingest markers.

The watcher and the web/CLI run as separate processes, so they can't share an
in-memory "currently ingesting" set. When the API/CLI ingests a file, it drops a
filesystem marker keyed by the file's vault-relative path; the watcher checks for
the marker and skips that file until the coordinator's ingest+compile cycle clears
it. This prevents the double-ingest (server-on-upload + watcher-on-file-appears).
"""

import hashlib
from pathlib import Path


def _dir() -> Path:
    d = Path.home() / ".pkms" / "ingesting"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _key(rel_path: str) -> Path:
    return _dir() / (hashlib.sha1(rel_path.encode("utf-8")).hexdigest() + ".marker")


def mark_ingesting(rel_path: str) -> None:
    try:
        _key(rel_path).write_text(rel_path, encoding="utf-8")
    except OSError:
        pass


def clear_ingesting(rel_path: str) -> None:
    try:
        _key(rel_path).unlink(missing_ok=True)
    except OSError:
        pass


def is_ingesting(rel_path: str) -> bool:
    return _key(rel_path).exists()
