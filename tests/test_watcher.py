import threading
import time
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pytest

from pkms.watcher import VaultWatcher, _VaultEventHandler, _hash_file

# ── minimal config fixture ────────────────────────────────────────────────────

BASE_CONFIG = {
    "watcher": {
        "enabled": True,
        "debounce_seconds": 0.05,  # short for tests
        "watch_paths": ["raw/"],
    },
    "ingest": {
        "supported_extensions": [".md", ".pdf", ".html", ".txt"],
    },
}

# ── _hash_file ────────────────────────────────────────────────────────────────

def test_hash_file_consistent(tmp_path):
    f = tmp_path / "doc.txt"
    f.write_bytes(b"hello world")
    h1 = _hash_file(str(f))
    h2 = _hash_file(str(f))
    assert h1 == h2
    assert h1.startswith("sha256:")


def test_hash_file_changes_on_content_change(tmp_path):
    f = tmp_path / "doc.txt"
    f.write_bytes(b"version 1")
    h1 = _hash_file(str(f))
    f.write_bytes(b"version 2")
    h2 = _hash_file(str(f))
    assert h1 != h2


def test_hash_file_returns_empty_for_missing_file():
    h = _hash_file("/nonexistent/path/file.txt")
    assert h == ""


# ── _VaultEventHandler._process ──────────────────────────────────────────────

def _make_handler(tmp_path, on_change, get_file_return=None):
    vault_root = str(tmp_path)
    db_path = str(tmp_path / ".search-index")
    with patch("pkms.watcher.get_file", return_value=get_file_return):
        handler = _VaultEventHandler(vault_root, db_path, BASE_CONFIG, on_change)
    return handler, vault_root


def test_process_triggers_callback_for_new_file(tmp_path):
    raw_dir = tmp_path / "vault" / "default" / "raw"
    raw_dir.mkdir(parents=True)
    f = raw_dir / "paper.pdf"
    f.write_bytes(b"%PDF content")

    callback = MagicMock()
    handler, _ = _make_handler(tmp_path, callback, get_file_return=None)

    with patch("pkms.watcher.get_file", return_value=None):
        handler._process(str(f))

    callback.assert_called_once()
    rel_path, file_hash = callback.call_args[0]
    assert "paper.pdf" in rel_path
    assert file_hash.startswith("sha256:")


def test_process_skips_unsupported_extension(tmp_path):
    raw_dir = tmp_path / "vault" / "default" / "raw"
    raw_dir.mkdir(parents=True)
    f = raw_dir / "file.zip"
    f.write_bytes(b"zipdata")

    callback = MagicMock()
    handler, _ = _make_handler(tmp_path, callback)
    handler._process(str(f))
    callback.assert_not_called()


def test_process_skips_unchanged_hash(tmp_path):
    raw_dir = tmp_path / "vault" / "default" / "raw"
    raw_dir.mkdir(parents=True)
    f = raw_dir / "doc.pdf"
    f.write_bytes(b"%PDF content")

    existing_hash = _hash_file(str(f))
    callback = MagicMock()
    handler, _ = _make_handler(tmp_path, callback)

    with patch("pkms.watcher.get_file", return_value={"hash": existing_hash}):
        handler._process(str(f))
    callback.assert_not_called()


def test_process_triggers_when_hash_changed(tmp_path):
    raw_dir = tmp_path / "vault" / "default" / "raw"
    raw_dir.mkdir(parents=True)
    f = raw_dir / "doc.pdf"
    f.write_bytes(b"%PDF new content")

    callback = MagicMock()
    handler, _ = _make_handler(tmp_path, callback)

    with patch("pkms.watcher.get_file", return_value={"hash": "sha256:oldhash"}):
        handler._process(str(f))
    callback.assert_called_once()


def test_process_skips_missing_file(tmp_path):
    callback = MagicMock()
    handler, _ = _make_handler(tmp_path, callback)
    handler._process(str(tmp_path / "vault" / "default" / "raw" / "ghost.pdf"))
    callback.assert_not_called()


def test_process_callback_exception_does_not_raise(tmp_path):
    raw_dir = tmp_path / "vault" / "default" / "raw"
    raw_dir.mkdir(parents=True)
    f = raw_dir / "doc.pdf"
    f.write_bytes(b"content")

    def bad_callback(path, h):
        raise RuntimeError("boom")

    handler, _ = _make_handler(tmp_path, bad_callback)
    with patch("pkms.watcher.get_file", return_value=None):
        handler._process(str(f))  # must not raise


# ── debounce ──────────────────────────────────────────────────────────────────

def test_debounce_fires_once_per_burst(tmp_path):
    raw_dir = tmp_path / "vault" / "default" / "raw"
    raw_dir.mkdir(parents=True)
    f = raw_dir / "doc.pdf"
    f.write_bytes(b"content")

    callback = MagicMock()
    handler = _VaultEventHandler(str(tmp_path), str(tmp_path / "idx"), BASE_CONFIG, callback)

    with patch("pkms.watcher.get_file", return_value=None):
        handler.start_worker()
        # Simulate 3 rapid events for the same file
        for _ in range(3):
            handler._record(str(f))
        time.sleep(0.3)  # wait past the debounce window
        handler.stop_worker()

    assert callback.call_count == 1


def test_sustained_events_flush_within_max_wait(tmp_path):
    # A stream of events that never goes quiet must still flush once the
    # max_batch_seconds latency bound is hit (the old Timer scheme reset the
    # debounce on every event and could defer processing forever).
    raw_dir = tmp_path / "vault" / "default" / "raw"
    raw_dir.mkdir(parents=True)
    f = raw_dir / "doc.pdf"
    f.write_bytes(b"content")

    config = {**BASE_CONFIG, "watcher": {**BASE_CONFIG["watcher"],
                                         "debounce_seconds": 0.1,
                                         "max_batch_seconds": 0.25}}
    callback = MagicMock()
    handler = _VaultEventHandler(str(tmp_path), str(tmp_path / "idx"), config, callback)

    with patch("pkms.watcher.get_file", return_value=None):
        handler.start_worker()
        deadline = time.monotonic() + 0.5
        while time.monotonic() < deadline and callback.call_count == 0:
            handler._record(str(f))   # keep the path "hot": debounce never elapses
            time.sleep(0.03)
        handler.stop_worker()

    assert callback.call_count >= 1


def test_stop_worker_is_idempotent_and_safe_without_start(tmp_path):
    handler = _VaultEventHandler(str(tmp_path), str(tmp_path / "idx"), BASE_CONFIG, MagicMock())
    handler.stop_worker()  # never started — must not raise
    handler.start_worker()
    handler.stop_worker()
    handler.stop_worker()  # double stop — must not raise


# ── VaultWatcher integration ──────────────────────────────────────────────────

def test_vault_watcher_disabled_does_not_start(tmp_path):
    config = {**BASE_CONFIG, "watcher": {**BASE_CONFIG["watcher"], "enabled": False}}
    (tmp_path / "vault" / "default" / "raw").mkdir(parents=True)

    with patch("pkms.watcher.Observer") as MockObserver:
        mock_observer = MockObserver.return_value
        watcher = VaultWatcher(str(tmp_path), str(tmp_path / "idx"), config, MagicMock())
        watcher.start()
        mock_observer.start.assert_not_called()


def test_vault_watcher_start_stop(tmp_path):
    (tmp_path / "vault" / "default" / "raw").mkdir(parents=True)

    with patch("pkms.watcher.Observer") as MockObserver:
        mock_observer = MockObserver.return_value
        mock_observer.is_alive.return_value = False

        watcher = VaultWatcher(str(tmp_path), str(tmp_path / "idx"), BASE_CONFIG, MagicMock())
        watcher.start()
        mock_observer.start.assert_called_once()
        watcher.stop()
        mock_observer.stop.assert_called_once()
        mock_observer.join.assert_called_once()


def test_vault_watcher_schedules_vault_root(tmp_path):
    (tmp_path / "vault" / "default" / "raw").mkdir(parents=True)

    with patch("pkms.watcher.Observer") as MockObserver:
        mock_observer = MockObserver.return_value
        VaultWatcher(str(tmp_path), str(tmp_path / "idx"), BASE_CONFIG, MagicMock())

    schedule_calls = mock_observer.schedule.call_args_list
    assert len(schedule_calls) == 1
    scheduled_path = schedule_calls[0][0][1]  # second positional arg is path
    assert scheduled_path.rstrip("/").endswith("vault")


def test_vault_watcher_creates_vault_dir_if_missing(tmp_path):
    vault_dir = tmp_path / "vault"
    assert not vault_dir.exists()

    with patch("pkms.watcher.Observer"):
        VaultWatcher(str(tmp_path), str(tmp_path / "idx"), BASE_CONFIG, MagicMock())

    assert vault_dir.exists()
