import io
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

import pkms.web as web_module
from pkms.web import app

CLIENT = TestClient(app, raise_server_exceptions=False)

VAULT_ROOT = "/tmp/test_vault_web"

INGEST_RESULT = {
    "status": "DONE",
    "path": "vault/raw/paper.pdf",
    "hash": "sha256:abc123",
    "qdrant_ids": ["id1"],
    "collection": "raw",
    "n_chunks": 5,
    "indexed_at": "2026-06-11T10:00:00Z",
}

QUERY_RESULT = {
    "answer_md": "Transformers use self-attention.",
    "sources": ["vault/wiki/articles/transformers.md"],
    "coverage": "full",
    "session_id": "query_20260611T100000_alice",
    "output_path": "vault/outputs/query_20260611T100000_alice.md",
}

CFG = {
    "ingest": {"auto_compile": False, "supported_extensions": [".pdf", ".md"]},
}

# ── /status ───────────────────────────────────────────────────────────────────

def test_status_returns_200(tmp_path):
    with patch.object(web_module, "_VAULT_ROOT", tmp_path):
        resp = CLIENT.get("/status")
    assert resp.status_code == 200
    body = resp.json()
    assert "vault_root" in body
    assert "db_exists" in body
    assert body["db_exists"] is False  # tmp_path has no .search-index


def test_status_db_exists_true(tmp_path):
    vault_dir = tmp_path / "vault"
    vault_dir.mkdir()
    (vault_dir / ".search-index").write_bytes(b"")
    (vault_dir / "raw").mkdir()
    (vault_dir / "wiki").mkdir()
    (vault_dir / "outputs").mkdir()
    with patch.object(web_module, "_VAULT_ROOT", tmp_path):
        resp = CLIENT.get("/status")
    body = resp.json()
    assert body["db_exists"] is True
    assert "default" in body["projects"]


# ── POST /query ───────────────────────────────────────────────────────────────

def test_query_happy_path(tmp_path):
    with patch.object(web_module, "_VAULT_ROOT", tmp_path), \
         patch.object(web_module, "_get_config", return_value=CFG), \
         patch("pkms.web.handle_query", return_value=QUERY_RESULT) as mock_q:
        resp = CLIENT.post("/query", json={"text": "What are transformers?", "user_id": "alice"})

    assert resp.status_code == 200
    body = resp.json()
    assert body["answer_md"] == "Transformers use self-attention."
    assert body["coverage"] == "full"
    assert body["sources"] == ["vault/wiki/articles/transformers.md"]
    mock_q.assert_called_once()


def test_query_passes_user_id(tmp_path):
    with patch.object(web_module, "_VAULT_ROOT", tmp_path), \
         patch.object(web_module, "_get_config", return_value=CFG), \
         patch("pkms.web.handle_query", return_value=QUERY_RESULT) as mock_q:
        CLIENT.post("/query", json={"text": "Q?", "user_id": "bob"})

    call_kwargs = mock_q.call_args[1]
    assert call_kwargs["user_id"] == "bob"


def test_query_passes_session_id(tmp_path):
    with patch.object(web_module, "_VAULT_ROOT", tmp_path), \
         patch.object(web_module, "_get_config", return_value=CFG), \
         patch("pkms.web.handle_query", return_value=QUERY_RESULT) as mock_q:
        CLIENT.post("/query", json={"text": "Q?", "user_id": "a", "session_id": "ses-42"})

    call_kwargs = mock_q.call_args[1]
    assert call_kwargs["session_id"] == "ses-42"


def test_query_empty_text_returns_422(tmp_path):
    resp = CLIENT.post("/query", json={"text": "", "user_id": "alice"})
    assert resp.status_code == 422


def test_query_missing_text_returns_422(tmp_path):
    resp = CLIENT.post("/query", json={"user_id": "alice"})
    assert resp.status_code == 422


def test_query_handler_exception_returns_500(tmp_path):
    with patch.object(web_module, "_VAULT_ROOT", tmp_path), \
         patch.object(web_module, "_get_config", return_value=CFG), \
         patch("pkms.web.handle_query", side_effect=RuntimeError("embed service down")):
        resp = CLIENT.post("/query", json={"text": "Q?", "user_id": "alice"})
    assert resp.status_code == 500


# ── POST /ingest (URL) ────────────────────────────────────────────────────────

def test_ingest_url_happy_path(tmp_path):
    with patch.object(web_module, "_VAULT_ROOT", tmp_path), \
         patch.object(web_module, "_get_config", return_value=CFG), \
         patch("pkms.web.handle_ingest", return_value=INGEST_RESULT) as mock_i:
        resp = CLIENT.post("/ingest", json={"url": "https://arxiv.org/abs/1234.5678"})

    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "DONE"
    assert body["n_chunks"] == 5
    mock_i.assert_called_once_with(
        "https://arxiv.org/abs/1234.5678", str(tmp_path), CFG, project="default"
    )


def test_ingest_url_exception_returns_500(tmp_path):
    with patch.object(web_module, "_VAULT_ROOT", tmp_path), \
         patch.object(web_module, "_get_config", return_value=CFG), \
         patch("pkms.web.handle_ingest", side_effect=ConnectionError("fetch failed")):
        resp = CLIENT.post("/ingest", json={"url": "https://example.com/paper.pdf"})
    assert resp.status_code == 500


def test_ingest_url_missing_field_returns_422(tmp_path):
    resp = CLIENT.post("/ingest", json={})
    assert resp.status_code == 422


# ── POST /ingest/file ─────────────────────────────────────────────────────────

def test_ingest_file_happy_path(tmp_path):
    (tmp_path / "vault" / "raw").mkdir(parents=True)
    with patch.object(web_module, "_VAULT_ROOT", tmp_path), \
         patch.object(web_module, "_get_config", return_value=CFG), \
         patch("pkms.web.handle_ingest", return_value=INGEST_RESULT) as mock_i:
        file_bytes = b"%PDF fake content"
        resp = CLIENT.post(
            "/ingest/file",
            files={"file": ("paper.pdf", io.BytesIO(file_bytes), "application/pdf")},
        )

    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "DONE"
    mock_i.assert_called_once()
    # Confirm the path passed to handle_ingest is under vault/raw/
    ingested_path = mock_i.call_args[0][0]
    assert "raw" in ingested_path
    assert "paper.pdf" in ingested_path


def test_ingest_file_exception_returns_500(tmp_path):
    (tmp_path / "vault" / "raw").mkdir(parents=True)
    with patch.object(web_module, "_VAULT_ROOT", tmp_path), \
         patch.object(web_module, "_get_config", return_value=CFG), \
         patch("pkms.web.handle_ingest", side_effect=RuntimeError("parse failed")):
        resp = CLIENT.post(
            "/ingest/file",
            files={"file": ("doc.pdf", io.BytesIO(b"content"), "application/pdf")},
        )
    assert resp.status_code == 500


def test_ingest_file_no_file_returns_422(tmp_path):
    resp = CLIENT.post("/ingest/file")
    assert resp.status_code == 422


# ── regression: upload must store a vault-relative key, not an absolute path ────
# An absolute key made the compiler's _project_of() fall back to "default" and
# misplace the compiled article while the DB recorded the correct project.

def test_ingest_file_stores_vault_relative_path(tmp_path):
    with patch.object(web_module, "_VAULT_ROOT", tmp_path), \
         patch.object(web_module, "_get_config", return_value=CFG), \
         patch("pkms.web.handle_ingest", return_value=INGEST_RESULT) as mock_i:
        resp = CLIENT.post(
            "/ingest/file",
            files={"file": ("paper.pdf", io.BytesIO(b"%PDF x"), "application/pdf")},
        )

    assert resp.status_code == 200
    ingested_path = mock_i.call_args[0][0]
    assert ingested_path == "vault/default/raw/paper.pdf"
    assert not Path(ingested_path).is_absolute()


def test_ingest_file_stores_path_under_requested_project(tmp_path):
    with patch.object(web_module, "_VAULT_ROOT", tmp_path), \
         patch.object(web_module, "_get_config", return_value=CFG), \
         patch("pkms.web.handle_ingest", return_value=INGEST_RESULT) as mock_i:
        resp = CLIENT.post(
            "/ingest/file",
            files={"file": ("paper.pdf", io.BytesIO(b"%PDF x"), "application/pdf")},
            data={"project": "og-mdai"},
        )

    assert resp.status_code == 200
    ingested_path = mock_i.call_args[0][0]
    assert ingested_path == "vault/og-mdai/raw/paper.pdf"
    # project kwarg and the path's project segment must agree
    assert mock_i.call_args[1]["project"] == "og-mdai"
