"""JSON API routes (roadmap 1.12 split): /query, /ingest, /ingest/file, /status.

Patched collaborators (handle_query, handle_ingest) are reached via `web.<name>`
at request time so patch("pkms.web.<name>") still targets them; everything else
is imported by name (deps compose by object identity, so dependency_overrides
works regardless of import path)."""
from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status

import pkms.web as web
from pkms.auth import AccessDenied
from pkms.guards import normalize_project
from pkms.web import (
    IngestResponse,
    IngestURLRequest,
    QueryRequest,
    QueryResponse,
    StatusResponse,
    _save_upload,
    get_config,
    get_current_user,
    get_projects,
    get_vault_root,
    logger,
)

router = APIRouter()


@router.post("/query", response_model=QueryResponse)
def query_endpoint(
    req: QueryRequest,
    config: dict = Depends(get_config),
    vault_root: str = Depends(get_vault_root),
    auth_user: str = Depends(get_current_user),
) -> QueryResponse:
    """Answer a question from the knowledge base."""
    try:
        result = web.handle_query(
            question=req.text,
            user_id=req.user_id,
            vault_root=vault_root,
            config=config,
            session_id=req.session_id,
            project=normalize_project(req.project),
            auth_user=auth_user,
        )
    except AccessDenied as exc:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))
    except Exception as exc:
        logger.error("Query failed: %s", exc)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc))

    return QueryResponse(
        answer_md=result["answer_md"],
        sources=result["sources"],
        coverage=result["coverage"],
        session_id=result["session_id"],
        output_path=result["output_path"],
    )


@router.post("/ingest", response_model=IngestResponse)
def ingest_url_endpoint(
    req: IngestURLRequest,
    config: dict = Depends(get_config),
    vault_root: str = Depends(get_vault_root),
    auth_user: str = Depends(get_current_user),
) -> IngestResponse:
    """Fetch and ingest a URL (arxiv PDF or web article)."""
    try:
        result = web.handle_ingest(req.url, vault_root, config,
                                   project=normalize_project(req.project), auth_user=auth_user)
    except AccessDenied as exc:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))
    except Exception as exc:
        logger.error("URL ingest failed: %s", exc)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc))

    return IngestResponse(
        status=result.get("status", "DONE"),
        path=result.get("path", ""),
        n_chunks=result.get("n_chunks", 0),
        hash=result.get("hash", ""),
    )


@router.post("/ingest/file", response_model=IngestResponse)
def ingest_file_endpoint(
    file: UploadFile = File(...),
    project: str = Form(default="default"),
    config: dict = Depends(get_config),
    vault_root: str = Depends(get_vault_root),
    auth_user: str = Depends(get_current_user),
) -> IngestResponse:
    """Upload a file and ingest it directly into vault/{project}/raw/."""
    project = normalize_project(project)
    rel_path = _save_upload(file, project, vault_root)

    try:
        result = web.handle_ingest(rel_path, vault_root, config, project=project,
                                   auth_user=auth_user)
    except AccessDenied as exc:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(exc))
    except Exception as exc:
        logger.error("File ingest failed: %s", exc)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc))

    return IngestResponse(
        status=result.get("status", "DONE"),
        path=result.get("path", ""),
        n_chunks=result.get("n_chunks", 0),
        hash=result.get("hash", ""),
    )


@router.get("/status", response_model=StatusResponse)
def status_endpoint(
    vault_root: str = Depends(get_vault_root),
    projects: list[str] = Depends(get_projects),
) -> StatusResponse:
    """Return vault health — existence of key directories and the .search-index."""
    vr = Path(vault_root)
    vault_dir = vr / "vault"
    return StatusResponse(
        vault_root=str(vr),
        db_exists=(vault_dir / ".search-index").exists(),
        projects=projects,
    )
