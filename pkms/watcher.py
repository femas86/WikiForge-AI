import hashlib
import logging
import threading
import time
from pathlib import Path
from typing import Any, Callable

from watchdog.events import FileCreatedEvent, FileModifiedEvent, FileSystemEventHandler
from watchdog.observers import Observer

from pkms.db import get_file
from pkms.guards import _PROJECT_NAME_RE

logger = logging.getLogger(__name__)

# ── hash helper ───────────────────────────────────────────────────────────────

def _hash_file(path: str) -> str:
    h = hashlib.sha256()
    try:
        with open(path, "rb") as fh:
            for block in iter(lambda: fh.read(65536), b""):
                h.update(block)
        return f"sha256:{h.hexdigest()}"
    except OSError:
        return ""


# ── debounced event handler ───────────────────────────────────────────────────

class _VaultEventHandler(FileSystemEventHandler):
    """Debounced handler for vault/raw/ changes.

    Events accumulate in `_pending`; a single worker thread flushes a path once
    it has been quiet for `debounce_seconds` — or unconditionally once it has
    waited `max_batch_seconds`, so a sustained event stream cannot defer
    processing forever. The one worker serializes `_process`, so a path can
    never be hash-checked and ingested by two threads at once (the old
    Timer-per-burst scheme could run overlapping flushes and double-ingest).
    Hash-checks each path against .search-index before calling back — skips
    files that haven't changed.
    """

    def __init__(
        self,
        vault_root: str,
        db_path: str,
        config: dict[str, Any],
        on_change: Callable[[str, str], None],
    ):
        super().__init__()
        self._vault_root = vault_root
        self._db_path = db_path
        self._config = config
        self._on_change = on_change
        self._debounce = config["watcher"]["debounce_seconds"]
        self._max_wait = config["watcher"].get("max_batch_seconds", self._debounce * 10)
        self._supported = set(config["ingest"]["supported_extensions"])
        # Per-project subdirs that trigger ingestion (e.g. {"raw"})
        self._watched_subdirs = {
            p.strip("/") for p in config["watcher"].get("watch_paths", ["raw/"])
        }
        # abs_path → (first-event ts, last-event ts)
        self._pending: dict[str, tuple[float, float]] = {}
        self._lock = threading.Lock()
        self._wake = threading.Event()
        self._stop_event = threading.Event()
        self._worker: threading.Thread | None = None

    # ── worker lifecycle ──────────────────────────────────────────────────────

    def start_worker(self) -> None:
        if self._worker is not None and self._worker.is_alive():
            return
        self._stop_event.clear()
        self._worker = threading.Thread(
            target=self._worker_loop, name="pkms-watcher-flush", daemon=True
        )
        self._worker.start()

    def stop_worker(self) -> None:
        self._stop_event.set()
        self._wake.set()
        if self._worker is not None:
            self._worker.join(timeout=5.0)
            self._worker = None

    def _record(self, abs_path: str) -> None:
        now = time.monotonic()
        with self._lock:
            first = self._pending.get(abs_path, (now, now))[0]
            self._pending[abs_path] = (first, now)
        self._wake.set()

    def _worker_loop(self) -> None:
        while not self._stop_event.is_set():
            with self._lock:
                snapshot = dict(self._pending)
            if not snapshot:
                self._wake.wait()
                self._wake.clear()
                continue
            now = time.monotonic()
            # A path is ripe when quiet for the debounce window, or when it has
            # been pending max_wait overall (latency bound under event storms).
            deadlines = {
                path: min(last + self._debounce, first + self._max_wait)
                for path, (first, last) in snapshot.items()
            }
            ripe = sorted(path for path, dl in deadlines.items() if dl <= now)
            if not ripe:
                self._wake.wait(timeout=max(0.01, min(deadlines.values()) - now))
                self._wake.clear()
                continue
            with self._lock:
                for path in ripe:
                    self._pending.pop(path, None)
            for path in ripe:
                self._process(path)

    def _process(self, abs_path: str) -> None:
        p = Path(abs_path)
        if p.suffix.lower() not in self._supported:
            logger.debug("Watcher: skipping unsupported extension %s", abs_path)
            return
        if not p.exists():
            logger.debug("Watcher: file gone before processing %s", abs_path)
            return

        new_hash = _hash_file(abs_path)
        if not new_hash:
            return

        # Vault-relative path: e.g. vault/{project}/raw/paper.pdf
        vault_dir = Path(self._vault_root) / "vault"
        try:
            rel = str(p.relative_to(vault_dir.parent))
        except ValueError:
            logger.debug("Watcher: path outside vault %s — skipping", abs_path)
            return

        # Only react inside {project}/{watched-subdir}/ (skips wiki/, outputs/,
        # .search-index sidecars, and invalid project dirs)
        parts = Path(rel).parts
        if (
            len(parts) < 4
            or parts[0] != "vault"
            or not _PROJECT_NAME_RE.match(parts[1])
            or parts[2] not in self._watched_subdirs
        ):
            logger.debug("Watcher: %s not in a watched project subdir — skipping", rel)
            return

        existing = get_file(self._db_path, rel)
        if existing and existing["hash"] == new_hash:
            logger.debug("Watcher: hash unchanged for %s — skipping", rel)
            return

        logger.info("Watcher: change detected %s (hash=%s)", rel, new_hash[:16])
        try:
            self._on_change(rel, new_hash)
        except Exception as exc:
            logger.error("Watcher: on_change callback failed for %s: %s", rel, exc)

    # watchdog event overrides

    def on_created(self, event) -> None:
        if not event.is_directory:
            self._record(event.src_path)

    def on_modified(self, event) -> None:
        if not event.is_directory:
            self._record(event.src_path)


# ── public API ────────────────────────────────────────────────────────────────

class VaultWatcher:
    """Watches every project's raw/ for file changes; fires on_change(vault_rel_path, hash).

    The whole vault root is observed (so projects created after startup are
    covered); the handler filters events to {project}/{watched-subdir}/ paths.
    """

    def __init__(
        self,
        vault_root: str,
        db_path: str,
        config: dict[str, Any],
        on_change: Callable[[str, str], None],
    ):
        self._vault_root = vault_root
        self._config = config
        self._handler = _VaultEventHandler(vault_root, db_path, config, on_change)
        self._observer = Observer()

        vault_dir = Path(vault_root) / "vault"
        vault_dir.mkdir(parents=True, exist_ok=True)
        self._observer.schedule(self._handler, str(vault_dir), recursive=True)
        logger.info("Watcher: monitoring %s (all projects)", vault_dir)

    def start(self) -> None:
        if not self._config["watcher"].get("enabled", True):
            logger.info("Watcher: disabled in config — not starting")
            return
        self._handler.start_worker()
        self._observer.start()
        logger.info("Watcher: started")

    def stop(self) -> None:
        self._observer.stop()
        self._observer.join()
        self._handler.stop_worker()
        logger.info("Watcher: stopped")

    def is_alive(self) -> bool:
        return self._observer.is_alive()
