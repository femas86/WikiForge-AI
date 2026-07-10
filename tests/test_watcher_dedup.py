from unittest.mock import MagicMock, patch

import pkms.ingest_marker as marker
import pkms.coordinator as coord
import pkms.ingestor as ingestor

CFG = {"ingest": {"auto_compile": False}}


# ── ingest marker (cross-process dedup) ───────────────────────────────────────

def test_marker_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setattr(marker, "_dir", lambda: tmp_path)
    rel = "vault/default/raw/x.pdf"
    assert not marker.is_ingesting(rel)
    marker.mark_ingesting(rel)
    assert marker.is_ingesting(rel)
    marker.clear_ingesting(rel)
    assert not marker.is_ingesting(rel)


# ── handle_ingest marks the file for the whole cycle, then clears ─────────────

def test_handle_ingest_marks_and_clears(tmp_path):
    with patch.object(coord, "_ensure_db", return_value="db"), \
         patch.object(coord, "ingest", return_value={"status": "DONE", "path": "vault/default/raw/x.pdf"}), \
         patch.object(coord, "_write_ingest_result"), \
         patch.object(coord, "handle_compile"), \
         patch.object(coord, "mark_ingesting") as mk, \
         patch.object(coord, "clear_ingesting") as ck:
        coord.handle_ingest("vault/default/raw/x.pdf", str(tmp_path), CFG, project="default")
    mk.assert_called_once_with("vault/default/raw/x.pdf")
    ck.assert_called_once_with("vault/default/raw/x.pdf")


def test_handle_ingest_clears_marker_even_on_error(tmp_path):
    with patch.object(coord, "_ensure_db", return_value="db"), \
         patch.object(coord, "ingest", side_effect=RuntimeError("boom")), \
         patch.object(coord, "mark_ingesting"), \
         patch.object(coord, "clear_ingesting") as ck:
        try:
            coord.handle_ingest("vault/default/raw/x.pdf", str(tmp_path), CFG, project="default")
        except RuntimeError:
            pass
    ck.assert_called_once_with("vault/default/raw/x.pdf")


# ── URL ingest marks the file BEFORE it lands on disk (regression) ────────────

def test_fetch_and_ingest_marks_before_write(tmp_path, monkeypatch):
    """The watcher fires on the file appearing; if the marker is set only after
    the fetch+ingest (the old bug), the watcher double-ingests. Guard that the
    marker exists by the time the file is written / ingest runs."""
    monkeypatch.setattr(marker, "_dir", lambda: tmp_path)

    resp = MagicMock()
    resp.content = b"%PDF-1.4 fake"
    resp.headers = {"content-type": "application/pdf"}
    resp.raise_for_status = lambda: None

    seen = {}

    def fake_ingest(path, vault_root, db_path, config, project="default", force=False):
        # By the time ingest runs the file is already on disk → marker MUST exist.
        seen["marked_at_ingest"] = marker.is_ingesting(path)
        seen["path"] = path
        return {"status": "DONE", "path": path, "hash": "h", "qdrant_ids": [],
                "collection": "raw", "indexed_at": "t"}

    with patch.object(ingestor.httpx, "get", return_value=resp), \
         patch.object(ingestor, "ingest", side_effect=fake_ingest):
        result = ingestor.fetch_and_ingest(
            "https://arxiv.org/pdf/1810.04805", str(tmp_path), "db",
            {"fetch": {}}, project="default")

    assert seen["marked_at_ingest"] is True          # marker present before/at ingest
    assert marker.is_ingesting(seen["path"])         # still marked after (coordinator clears it)


def test_fetch_and_ingest_clears_marker_on_ingest_error(tmp_path, monkeypatch):
    """A failed URL ingest must not leave an orphan marker that blocks the watcher forever."""
    monkeypatch.setattr(marker, "_dir", lambda: tmp_path)

    resp = MagicMock()
    resp.content = b"%PDF-1.4 fake"
    resp.headers = {"content-type": "application/pdf"}
    resp.raise_for_status = lambda: None

    with patch.object(ingestor.httpx, "get", return_value=resp), \
         patch.object(ingestor, "ingest", side_effect=RuntimeError("boom")):
        try:
            ingestor.fetch_and_ingest(
                "https://arxiv.org/pdf/1810.04805", str(tmp_path), "db",
                {"fetch": {}}, project="default")
        except RuntimeError:
            pass

    assert not marker.is_ingesting("vault/default/raw/1810.04805.pdf")


# ── reconciliation gate ───────────────────────────────────────────────────────

def test_reconcile_compiles_only_projects_with_pending(tmp_path):
    with patch.object(coord, "_ensure_db", return_value="db"), \
         patch.object(coord, "list_projects", return_value=["p1", "p2"]), \
         patch.object(coord, "_get_uncompiled_raw_paths",
                      side_effect=lambda db, p: ["vault/p1/raw/a.pdf"] if p == "p1" else []), \
         patch.object(coord, "handle_compile") as mc:
        total = coord.reconcile_projects(str(tmp_path), {"ingest": {}})
    assert total == 1
    assert mc.call_count == 1
    assert mc.call_args.kwargs["project"] == "p1"
