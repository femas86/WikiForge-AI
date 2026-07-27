import logging
import os
import re
import shutil
import sqlite3
import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent.parent / ".env")

import httpx
import uvicorn
from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, UploadFile, status
from fastapi.responses import HTMLResponse, RedirectResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from markupsafe import escape
from pydantic import BaseModel, Field

from pkms.compiler import (
    _extract_frontmatter_field,
    _extract_frontmatter_tags,
    _get_uncompiled_raw_paths,
)
from pkms.auth import ROLES, AccessDenied, current_user
from pkms.coordinator import (
    check_access,
    handle_compile,
    handle_ingest,
    handle_lint,
    handle_member,
    handle_query,
    handle_reindex,
    handle_remove_doc,
    _watcher_ready_path,
)
from pkms.db import init_db
# Link-graph domain logic lives in pkms.graph (roadmap 1.12 stage 1); re-exported
# here so the existing import surface (pkms.web._wiki_link_graph, used by tests
# and the ui router) is preserved after the route split.
from pkms.graph import _norm_concept, _wiki_link_graph
from pkms.events import astream, create_job, install_handler, list_active, run_job
from pkms.guards import guard_write, list_projects, load_config, normalize_project, validate_project
from pkms.linter import _extract_wiki_links
from pkms.lock import LockTimeout
from pkms.qdrant_store import ensure_collections
from pkms.user_prefs import list_users, load_user_style, save_user_style

logger = logging.getLogger(__name__)

# NB: this module is pkms/web/__init__.py, so parent.parent is the pkms/ package
# dir and parent.parent.parent is the repo root (where pkms.config.yaml / vault live).
_CONFIG_PATH = Path(__file__).parent.parent.parent / "pkms.config.yaml"
_VAULT_ROOT = Path(__file__).parent.parent.parent


def _recover_uncompiled(config: dict) -> None:
    """Startup recovery sweep: compile any raw docs left uncompiled by an
    interrupted ingest/auto-compile (e.g. a server restart killed the job).

    Runs in a background thread (non-blocking), sequentially per project, and is
    entirely non-fatal — if services are down it just logs and gives up; the next
    restart sweeps again. Self-healing across restarts.
    """
    try:
        # The watcher already reconciles raw↔wiki before it signals ready and owns
        # ongoing reconciliation. If it's up, skip: otherwise both processes compile
        # the same uncompiled docs on startup (double work, and a doubled retry storm
        # when a doc can't compile). This recovery is the fallback for a no-watcher run.
        if _watcher_ready_path().exists():
            logger.info("Recovery: watcher is running and owns reconciliation — skipping")
            return
        vault_root = _vault_root()
        db = str(Path(vault_root) / "vault" / ".search-index")
        if not Path(db).exists():
            return
        for project in _projects():
            try:
                pending = _get_uncompiled_raw_paths(db, project)
            except Exception as exc:
                logger.warning("Recovery: pending lookup failed for %s: %s", project, exc)
                continue
            if not pending:
                continue
            logger.info("Recovery: %d uncompiled doc(s) in project %s — compiling", len(pending), project)
            try:
                handle_compile(scope="all", vault_root=vault_root, config=config, project=project)
            except Exception as exc:
                logger.warning("Recovery compile failed for %s: %s", project, exc)
    except Exception as exc:
        logger.warning("Startup recovery sweep failed: %s", exc)


@asynccontextmanager
async def _lifespan(app: FastAPI):
    config = _get_config()
    install_handler()  # bridge pkms.* logs into per-job SSE queues
    # Initialise SQLite .search-index
    db = str(_VAULT_ROOT / "vault" / ".search-index")
    try:
        init_db(db)
    except Exception as exc:
        logger.warning("DB init failed at startup: %s", exc)
    # Ensure Qdrant collections exist
    try:
        ensure_collections(config)
    except Exception as exc:
        logger.warning("Qdrant not reachable at startup: %s — collections will be created on first use", exc)
    # Recover orphaned raw left uncompiled by an interrupted job (background, non-blocking)
    if config.get("compile", {}).get("recover_on_startup", True):
        threading.Thread(
            target=_recover_uncompiled, args=(config,), daemon=True, name="recover-uncompiled"
        ).start()
    yield


app = FastAPI(title="PKMS Web API", version="0.1.0", lifespan=_lifespan)

_TEMPLATES_DIR = Path(__file__).parent.parent / "templates"
_STATIC_DIR    = Path(__file__).parent.parent / "static"
templates = Jinja2Templates(directory=str(_TEMPLATES_DIR))
app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")

# ── lazy singletons ───────────────────────────────────────────────────────────

_config: dict[str, Any] | None = None


def _get_config() -> dict[str, Any]:
    global _config
    if _config is None:
        _config = load_config(str(_CONFIG_PATH))
    return _config


def _vault_root() -> str:
    return str(_VAULT_ROOT)


def _projects(vault_root: str | None = None) -> list[str]:
    root = vault_root if vault_root is not None else _vault_root()
    projects = list_projects(Path(root) / "vault")
    if "default" not in projects:
        projects.insert(0, "default")
    return projects


_WIKILINK_RE = re.compile(r"\[\[([^\]|]+)(?:\|([^\]]*))?\]\]")
_SLUG_RE = re.compile(r"^[\w.-]+$")
_FRONTMATTER_RE = re.compile(r"^---\n.*?\n---\n?", re.DOTALL)


def _rewrite_wikilinks(md: str, project: str) -> str:
    """Turn [[slug]] / [[slug|anchor]] into markdown links to /wiki/{project}/{slug}."""
    def _sub(match: re.Match) -> str:
        target = match.group(1).strip()
        anchor = (match.group(2) or target).strip()
        slug = Path(target).stem if "/" in target else target
        return f"[{anchor}](/wiki/{project}/{slug})"
    return _WIKILINK_RE.sub(_sub, md)


def _save_upload(file: UploadFile, project: str, vault_root: str) -> str:
    """Persist an uploaded file into vault/{project}/raw/, rejecting path traversal.

    Returns the *vault-relative* key (vault/{project}/raw/{name}), mirroring
    fetch_and_ingest. This is the canonical path handed to handle_ingest: passing
    the absolute path here is what made the compiler's _project_of() fall back to
    "default" and misplace compiled articles.
    """
    try:
        validate_project(project)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))
    vault_dir = Path(vault_root) / "vault"
    raw_dir = vault_dir / project / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    # Browsers send a bare filename, but the multipart field is client-controlled
    name = Path(file.filename or "upload.bin").name
    if not name or name in {".", ".."}:
        name = "upload.bin"
    dest = raw_dir / name
    try:
        dest.resolve().relative_to(raw_dir.resolve())
    except ValueError:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid filename")

    try:
        guard_write("ingestor", str(dest), str(vault_dir))
    except PermissionError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))

    try:
        with dest.open("wb") as fh:
            shutil.copyfileobj(file.file, fh)
    except OSError as exc:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc))
    finally:
        file.file.close()
    return str(dest.relative_to(Path(vault_root)))


def _note_slug(title: str) -> str:
    """Filesystem-safe slug for a synthetic note filename."""
    slug = re.sub(r"[^\w\-]+", "_", title.strip().lower()).strip("_")[:60]
    return slug or "note"


def _save_note(question: str, answer_md: str, sources: str, project: str, title: str,
               vault_root: str) -> str:
    """Write an answer as a synthetic note into vault/{project}/raw/notes/.

    Returns the vault-relative key (vault/{project}/raw/notes/{slug}.md), mirroring
    _save_upload. The note re-enters the knowledge base via the normal ingest+compile
    path, so it must live under raw/ (ingestor's write boundary).
    """
    try:
        validate_project(project)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))
    vault_dir = Path(vault_root) / "vault"
    notes_dir = vault_dir / project / "raw" / "notes"
    notes_dir.mkdir(parents=True, exist_ok=True)

    dest = notes_dir / f"{_note_slug(title)}.md"
    try:
        dest.resolve().relative_to(notes_dir.resolve())
    except ValueError:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid note title")

    body = (
        f"# {title.strip() or 'Note'}\n\n"
        f"**Domanda:** {question.strip()}\n\n"
        f"**Risposta:**\n\n{answer_md.strip()}\n\n"
        f"**Fonti:** {sources.strip() or '(none)'}\n"
    )
    try:
        guard_write("ingestor", str(dest), str(vault_dir))
    except PermissionError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))
    dest.write_text(body, encoding="utf-8")
    return str(dest.relative_to(Path(vault_root)))


def _render_ingest_result(result: dict | None, error: Exception | None, config: dict) -> str:
    template = templates.get_template("ingest_result.html")
    if error is not None:
        return template.render({
            "error": str(error),
            "ingest_status": None, "path": "", "n_chunks": 0, "auto_compile": False,
        })
    result = result or {}
    return template.render({
        "ingest_status": result.get("status"),
        "path":          result.get("path", ""),
        "n_chunks":      result.get("n_chunks", 0),
        "auto_compile":  config["ingest"].get("auto_compile", False),
        "error":         None,
    })


_ingest_executor: ThreadPoolExecutor | None = None
_ingest_executor_lock = threading.Lock()


def _get_ingest_executor() -> ThreadPoolExecutor:
    """Bounded pool that serializes ingests (Ollama is a single CPU-bound resource
    shared across all projects). Excess jobs queue here instead of thrashing Ollama
    and then contending the per-project compile lock. Size via ingest.max_concurrent
    (default 1)."""
    global _ingest_executor
    if _ingest_executor is None:
        with _ingest_executor_lock:
            if _ingest_executor is None:
                n = max(1, int(_get_config().get("ingest", {}).get("max_concurrent", 1)))
                _ingest_executor = ThreadPoolExecutor(max_workers=n, thread_name_prefix="ingest")
    return _ingest_executor


def _spawn_ingest_job(target: str, project: str, config: dict, vault_root: str) -> str:
    """Submit handle_ingest to the bounded ingest pool; returns the job id.

    The job is created as "queued"; run_job flips it to "running" when a worker
    slot frees up. The UI re-attach panel shows queued jobs as waiting.
    """
    display = target if target.startswith(("http://", "https://")) else Path(target).name
    job_id = create_job(target=f"{display} → {project}")
    _get_ingest_executor().submit(
        run_job,
        job_id,
        lambda: handle_ingest(target, vault_root, config, project=project),
        lambda result, error: _render_ingest_result(result, error, config),
    )
    return job_id


def _render_reindex_result(result: dict | None, error: Exception | None) -> str:
    if error is not None:
        return f'<div class="alert alert-error"><strong>Reindex failed:</strong> {escape(str(error))}</div>'
    result = result or {}
    if result.get("status") == "SKIPPED":
        return '<div class="alert alert-warn">Nothing to reindex — the project has no raw documents.</div>'
    failed = result.get("failed") or []
    base = (
        f'Project <code>{escape(str(result.get("project","")))}</code>: '
        f'{result.get("reingested", 0)} document(s) re-embedded, '
        f'{result.get("articles_rebuilt", 0)} article(s) rebuilt (grounded).'
    )
    if failed:
        names = ", ".join(escape(p.rsplit("/", 1)[-1]) for p in failed)
        return (
            f'<div class="alert alert-warn"><strong>Reindex finished with {len(failed)} '
            f'failure(s).</strong> {base} Not rebuilt: <code>{names}</code>. '
            'They stay uncompiled — retry with a compile once the rate limit clears.</div>'
        )
    return f'<div class="alert alert-success"><strong>Reindex complete.</strong> {base}</div>'


def _spawn_reindex_job(project: str, config: dict, vault_root: str) -> str:
    """Submit handle_reindex to the ingest pool; returns the job id (streamed like ingest)."""
    job_id = create_job(target=f"reindex → {project}")
    _get_ingest_executor().submit(
        run_job,
        job_id,
        lambda: handle_reindex(project, vault_root, config),
        lambda result, error: _render_reindex_result(result, error),
    )
    return job_id


# ── request / response models ─────────────────────────────────────────────────

class QueryRequest(BaseModel):
    text: str = Field(..., min_length=1, description="The question to answer")
    user_id: str = Field(default="default", description="User ID for Mem0 personalisation")
    session_id: str | None = Field(default=None, description="Optional session ID to resume")
    project: str = Field(default="default", description="Project to query")


class QueryResponse(BaseModel):
    answer_md: str
    sources: list[str]
    coverage: str
    session_id: str
    output_path: str


class IngestURLRequest(BaseModel):
    url: str = Field(..., description="http(s):// URL to fetch and ingest (PDF or HTML)")
    project: str = Field(default="default", description="Project to ingest into")


class IngestResponse(BaseModel):
    status: str
    path: str
    n_chunks: int
    hash: str


class StatusResponse(BaseModel):
    vault_root: str
    db_exists: bool
    projects: list[str]


# ── endpoints ─────────────────────────────────────────────────────────────────

# ── FastAPI dependency seam (DI, roadmap 1.12/B4) ───────────────────────────────
# The AUTHORIZED caller is resolved from the trusted reverse-proxy/SSO header
# (see pkms.auth.current_user via get_current_user), never from a client-supplied
# body/cookie user.
# Endpoints depend on these instead of reading the module globals directly, so
# tests override them via app.dependency_overrides (see _override_deps in the web
# tests) rather than monkeypatching module state, and the B4 identity is a
# first-class injected dependency. The default implementations delegate to the
# lazy singletons, so behaviour is unchanged when nothing is overridden.
def get_config() -> dict[str, Any]:
    return _get_config()


def get_vault_root() -> str:
    return _vault_root()


def get_projects(vault_root: str = Depends(get_vault_root)) -> list[str]:
    return _projects(vault_root)


def get_current_user(request: Request, config: dict[str, Any] = Depends(get_config)) -> str:
    """The authorized caller, resolved from the trusted reverse-proxy/SSO header
    (never a client-supplied body/cookie). This is the B4 authorization identity.

    Sub-depends on get_config/get_vault_root so overriding those in tests (or
    swapping the header source in deployment) flows through automatically."""
    return current_user(request.headers, config)


def _forbidden_fragment(exc: Exception) -> HTMLResponse:
    return HTMLResponse(
        f'<div class="alert alert-error"><strong>Forbidden:</strong> {escape(str(exc))}</div>',
        status_code=status.HTTP_403_FORBIDDEN,
    )


# ── route modules (split out of this file, roadmap 1.12) ────────────────────────
# Imported LAST, after all shared infra above (app, deps, models, helpers, and the
# coordinator collaborators) is defined. The routers reference the test-patched
# names (handle_query, handle_ingest, _spawn_ingest_job, httpx, os,
# _get_uncompiled_raw_paths) via `import pkms.web as web` and call them as
# `web.<name>` at request time, so patch("pkms.web.<name>") still targets them.
from pkms.web import api as _api_routes  # noqa: E402
from pkms.web import ui as _ui_routes  # noqa: E402
from pkms.web import jobs as _jobs_routes  # noqa: E402
app.include_router(_api_routes.router)
app.include_router(_ui_routes.router)
app.include_router(_jobs_routes.router)


# ── dev server entry point ────────────────────────────────────────────────────

def serve(host: str | None = None, port: int | None = None, reload: bool = False) -> None:
    host = host or os.environ.get("PKMS_HOST", "0.0.0.0")
    port = port or int(os.environ.get("PKMS_PORT", "8000"))
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    uvicorn.run("pkms.web:app", host=host, port=port, reload=reload)


if __name__ == "__main__":
    serve()
