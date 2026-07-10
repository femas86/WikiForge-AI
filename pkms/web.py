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
load_dotenv(Path(__file__).parent.parent / ".env")

import httpx
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile, status
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
from pkms.coordinator import (
    handle_compile,
    handle_ingest,
    handle_lint,
    handle_query,
    handle_reindex,
    handle_remove_doc,
    _watcher_ready_path,
)
from pkms.db import init_db
from pkms.events import create_job, install_handler, list_active, run_job, stream
from pkms.guards import guard_write, list_projects, load_config, normalize_project, validate_project
from pkms.linter import _extract_wiki_links
from pkms.lock import LockTimeout
from pkms.qdrant_store import ensure_collections
from pkms.user_prefs import list_users, load_user_style, save_user_style

logger = logging.getLogger(__name__)

_CONFIG_PATH = Path(__file__).parent.parent / "pkms.config.yaml"
_VAULT_ROOT = Path(__file__).parent.parent


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

_TEMPLATES_DIR = Path(__file__).parent / "templates"
_STATIC_DIR    = Path(__file__).parent / "static"
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


def _projects() -> list[str]:
    projects = list_projects(Path(_vault_root()) / "vault")
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


def _save_upload(file: UploadFile, project: str) -> str:
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
    vault_dir = Path(_vault_root()) / "vault"
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
    return str(dest.relative_to(Path(_vault_root())))


def _note_slug(title: str) -> str:
    """Filesystem-safe slug for a synthetic note filename."""
    slug = re.sub(r"[^\w\-]+", "_", title.strip().lower()).strip("_")[:60]
    return slug or "note"


def _save_note(question: str, answer_md: str, sources: str, project: str, title: str) -> str:
    """Write an answer as a synthetic note into vault/{project}/raw/notes/.

    Returns the vault-relative key (vault/{project}/raw/notes/{slug}.md), mirroring
    _save_upload. The note re-enters the knowledge base via the normal ingest+compile
    path, so it must live under raw/ (ingestor's write boundary).
    """
    try:
        validate_project(project)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))
    vault_dir = Path(_vault_root()) / "vault"
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
    return str(dest.relative_to(Path(_vault_root())))


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


def _spawn_ingest_job(target: str, project: str, config: dict) -> str:
    """Submit handle_ingest to the bounded ingest pool; returns the job id.

    The job is created as "queued"; run_job flips it to "running" when a worker
    slot frees up. The UI re-attach panel shows queued jobs as waiting.
    """
    display = target if target.startswith(("http://", "https://")) else Path(target).name
    job_id = create_job(target=f"{display} → {project}")
    _get_ingest_executor().submit(
        run_job,
        job_id,
        lambda: handle_ingest(target, _vault_root(), config, project=project),
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


def _spawn_reindex_job(project: str, config: dict) -> str:
    """Submit handle_reindex to the ingest pool; returns the job id (streamed like ingest)."""
    job_id = create_job(target=f"reindex → {project}")
    _get_ingest_executor().submit(
        run_job,
        job_id,
        lambda: handle_reindex(project, _vault_root(), config),
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

@app.post("/query", response_model=QueryResponse)
def query_endpoint(req: QueryRequest) -> QueryResponse:
    """Answer a question from the knowledge base."""
    config = _get_config()
    try:
        result = handle_query(
            question=req.text,
            user_id=req.user_id,
            vault_root=_vault_root(),
            config=config,
            session_id=req.session_id,
            project=normalize_project(req.project),
        )
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


@app.post("/ingest", response_model=IngestResponse)
def ingest_url_endpoint(req: IngestURLRequest) -> IngestResponse:
    """Fetch and ingest a URL (arxiv PDF or web article)."""
    config = _get_config()
    try:
        result = handle_ingest(req.url, _vault_root(), config, project=normalize_project(req.project))
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


@app.post("/ingest/file", response_model=IngestResponse)
def ingest_file_endpoint(
    file: UploadFile = File(...),
    project: str = Form(default="default"),
) -> IngestResponse:
    """Upload a file and ingest it directly into vault/{project}/raw/."""
    config = _get_config()
    project = normalize_project(project)
    rel_path = _save_upload(file, project)

    try:
        result = handle_ingest(rel_path, _vault_root(), config, project=project)
    except Exception as exc:
        logger.error("File ingest failed: %s", exc)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc))

    return IngestResponse(
        status=result.get("status", "DONE"),
        path=result.get("path", ""),
        n_chunks=result.get("n_chunks", 0),
        hash=result.get("hash", ""),
    )


@app.get("/status", response_model=StatusResponse)
def status_endpoint() -> StatusResponse:
    """Return vault health — existence of key directories and the .search-index."""
    vr = Path(_vault_root())
    vault_dir = vr / "vault"
    return StatusResponse(
        vault_root=str(vr),
        db_exists=(vault_dir / ".search-index").exists(),
        projects=_projects(),
    )


# ── UI page routes ────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
def root():
    return RedirectResponse(url="/query")


@app.get("/query", response_class=HTMLResponse)
def ui_query_page(request: Request):
    current_user = request.cookies.get("pkms_user") or "default"
    users = sorted(set(list_users()) | {current_user})  # current user always selectable
    return templates.TemplateResponse(request, "query.html", {
        "active":       "query",
        "projects":     _projects(),
        "users":        users,
        "current_user": current_user,
    })


@app.get("/ingest", response_class=HTMLResponse)
def ui_ingest_page(request: Request):
    return templates.TemplateResponse(request, "ingest.html", {"active": "ingest", "projects": _projects()})


@app.get("/settings", response_class=HTMLResponse)
def ui_settings_page(request: Request, user_id: str | None = None):
    """Per-user answer-style settings (customizes querier output, not the wiki).

    The user is remembered via the `pkms_user` cookie, so returning via the nav
    link (which carries no query string) still shows that user's saved style.
    An explicit ?user_id= overrides the cookie.
    """
    uid = user_id or request.cookies.get("pkms_user") or "default"
    users = sorted(set(list_users()) | {uid})  # current user always selectable
    return templates.TemplateResponse(request, "settings.html", {
        "active":  "settings",
        "user_id": uid,
        "style":   load_user_style(uid),
        "users":   users,
    })


@app.get("/ui/status", response_class=HTMLResponse)
def ui_status_page(request: Request):
    config = _get_config()
    vr = Path(_vault_root())
    vault_dir = vr / "vault"

    # Probe Qdrant
    qdrant_ok = False
    try:
        host = config["qdrant"]["host"]
        port = config["qdrant"]["port"]
        r = httpx.get(f"http://{host}:{port}/healthz", timeout=2)
        qdrant_ok = r.status_code == 200
    except Exception:
        pass

    # Probe Ollama
    ollama_ok = False
    try:
        ollama_cfg = config.get("ollama", {})
        host = ollama_cfg.get("host", "localhost")
        port = ollama_cfg.get("port", 11434)
        r = httpx.get(f"http://{host}:{port}/api/tags", timeout=2)
        ollama_ok = r.status_code == 200
    except Exception:
        pass

    # Indexed docs from .search-index, per project
    raw_count = wiki_count = 0
    counts: dict[str, dict[str, int]] = {}
    db = vault_dir / ".search-index"
    if db.exists():
        try:
            conn = sqlite3.connect(str(db))
            for proj, coll, n in conn.execute(
                "SELECT project, collection, COUNT(*) FROM files GROUP BY project, collection"
            ):
                counts.setdefault(proj, {"raw": 0, "wiki": 0})
                if coll in ("raw", "wiki"):
                    counts[proj][coll] = n
            conn.close()
        except Exception:
            pass
        raw_count = sum(c["raw"] for c in counts.values())
        wiki_count = sum(c["wiki"] for c in counts.values())

    projects = [
        {"name": p,
         "raw": counts.get(p, {}).get("raw", 0),
         "wiki": counts.get(p, {}).get("wiki", 0)}
        for p in _projects()
    ]

    return templates.TemplateResponse(request, "status.html", {
        "active":      "status",
        "vault_root":  str(vr),
        "projects":    projects,
        "db":          db.exists(),
        "qdrant_ok":   qdrant_ok,
        "ollama_ok":   ollama_ok,
        "api_key_set": bool(os.environ.get("ANTHROPIC_API_KEY")),
        "raw_count":   raw_count,
        "wiki_count":  wiki_count,
    })


# Link-graph domain logic lives in pkms.graph (roadmap 1.12); re-exported here
# for the existing import surface (tests, callers).
from pkms.graph import _norm_concept, _wiki_link_graph  # noqa: E402


@app.get("/wiki/graph/data")
def ui_wiki_graph_data(project: str = "default", concepts: bool = True, tags: bool = True):
    """JSON link graph for a project. `concepts`/`tags` toggle those node types."""
    project = normalize_project(project)
    validate_project(project)
    vault_dir = Path(_vault_root()) / "vault"
    return _wiki_link_graph(vault_dir, project, include_concepts=concepts, include_tags=tags)


@app.get("/wiki/graph", response_class=HTMLResponse)
def ui_wiki_graph_page(request: Request, project: str = "default"):
    """Force-directed graph view of a project's wiki."""
    project = normalize_project(project)
    try:
        validate_project(project)
    except ValueError:
        return RedirectResponse(url="/wiki")
    return templates.TemplateResponse(request, "wiki_graph.html", {
        "active":   "wiki",
        "project":  project,
        "projects": _projects(),
    })


@app.get("/wiki", response_class=HTMLResponse)
def ui_wiki_index(request: Request, project: str = "default", tag: str = ""):
    """Browse one project's compiled wiki articles (read-only)."""
    project = normalize_project(project)
    try:
        validate_project(project)
    except ValueError:
        return RedirectResponse(url="/wiki")
    vault_dir = Path(_vault_root()) / "vault"
    articles_dir = vault_dir / project / "wiki" / "articles"

    articles = []
    all_tags: set[str] = set()
    if articles_dir.exists():
        for md_file in sorted(articles_dir.glob("*.md")):
            try:
                md = md_file.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            tags = _extract_frontmatter_tags(md)
            all_tags.update(tags)
            articles.append({
                "slug":    md_file.stem,
                "title":   _extract_frontmatter_field(md, "title") or md_file.stem,
                "summary": _extract_frontmatter_field(md, "summary_1line"),
                "tags":    tags,
            })

    active_tag = tag.strip()
    if active_tag:
        articles = [a for a in articles if active_tag in a["tags"]]

    # Raw documents ingested but not yet compiled into any article
    pending: list[str] = []
    db = vault_dir / ".search-index"
    if db.exists():
        try:
            pending = _get_uncompiled_raw_paths(str(db), project)
        except Exception as exc:
            logger.warning("Pending-compile lookup failed: %s", exc)

    return templates.TemplateResponse(request, "wiki.html", {
        "active":     "wiki",
        "project":    project,
        "projects":   _projects(),
        "articles":   articles,
        "pending":    pending,
        "all_tags":   sorted(all_tags),
        "active_tag": active_tag,
    })


@app.get("/wiki/{project}/{slug}", response_class=HTMLResponse)
def ui_wiki_article(request: Request, project: str, slug: str):
    """Render one wiki article through the client-side markdown pipeline."""
    project = normalize_project(project)
    try:
        validate_project(project)
    except ValueError:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Unknown project")
    if not _SLUG_RE.match(slug) or slug in {".", ".."}:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Article not found")

    articles_dir = Path(_vault_root()) / "vault" / project / "wiki" / "articles"
    article_file = articles_dir / f"{slug}.md"
    try:
        article_file.resolve().relative_to(articles_dir.resolve())
    except ValueError:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Article not found")
    if not article_file.exists():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Article not found")

    md = article_file.read_text(encoding="utf-8", errors="replace")
    title = _extract_frontmatter_field(md, "title") or slug
    tags = _extract_frontmatter_tags(md)
    date = _extract_frontmatter_field(md, "date")
    body = _rewrite_wikilinks(_FRONTMATTER_RE.sub("", md, count=1), project)

    # Provenance from the article_sources table (best effort)
    sources: list[str] = []
    db = Path(_vault_root()) / "vault" / ".search-index"
    if db.exists():
        try:
            conn = sqlite3.connect(str(db))
            wiki_path = f"vault/{project}/wiki/articles/{slug}.md"
            sources = [r[0] for r in conn.execute(
                "SELECT raw_path FROM article_sources WHERE wiki_path = ?", (wiki_path,)
            )]
            conn.close()
        except Exception:
            pass

    # Backlinks: other articles whose [[links]] point at this slug (reverse edges).
    # Only article→article edges can target a real slug, so sources are articles.
    graph = _wiki_link_graph(Path(_vault_root()) / "vault", project)
    labels = {n["id"]: n["label"] for n in graph["nodes"]}
    backlinks = [
        {"slug": e["source"], "title": labels.get(e["source"], e["source"])}
        for e in graph["links"] if e["target"] == slug
    ]

    return templates.TemplateResponse(request, "wiki_article.html", {
        "active":    "wiki",
        "project":   project,
        "slug":      slug,
        "title":     title,
        "tags":      tags,
        "date":      date,
        "body":      body,
        "sources":   sources,
        "backlinks": backlinks,
    })


# ── UI HTMX fragment routes ───────────────────────────────────────────────────

@app.post("/ui/query", response_class=HTMLResponse)
def ui_query_fragment(
    request: Request,
    text: str = Form(...),
    user_id: str = Form(default="default"),
    project: str = Form(default="default"),
):
    config = _get_config()
    try:
        result = handle_query(
            question=text,
            user_id=user_id,
            vault_root=_vault_root(),
            config=config,
            project=normalize_project(project),
        )
        return templates.TemplateResponse(request, "answer.html", {
            "answer_md": result["answer_md"],
            "sources":   result["sources"],
            "coverage":  result["coverage"],
            "question":  text,
            "project":   normalize_project(project),
        })
    except Exception as exc:
        logger.error("UI query failed: %s", exc)
        return HTMLResponse(
            f'<div class="alert alert-error"><strong>Error:</strong> {escape(exc)}</div>'
        )


@app.post("/ui/settings", response_class=HTMLResponse)
def ui_settings_save(
    request: Request,
    user_id: str = Form(default="default"),
    style: str = Form(default=""),
):
    """Persist a user's answer-style instructions; returns a confirmation fragment."""
    try:
        save_user_style(user_id, style)
        resp = HTMLResponse(
            f'<div class="alert alert-success">Saved answer style for '
            f'<strong>{escape(user_id)}</strong>.</div>'
        )
        # Remember the user so a later GET /settings (via the nav link, no query
        # string) reloads their saved style instead of the "default" user's.
        resp.set_cookie("pkms_user", user_id, max_age=60 * 60 * 24 * 365, samesite="lax")
        return resp
    except Exception as exc:
        logger.error("Settings save failed: %s", exc)
        return HTMLResponse(
            f'<div class="alert alert-error"><strong>Error:</strong> {escape(exc)}</div>'
        )


@app.post("/ui/promote", response_class=HTMLResponse)
def ui_promote_fragment(
    request: Request,
    question: str = Form(...),
    answer_md: str = Form(...),
    sources: str = Form(default=""),
    title: str = Form(default=""),
    project: str = Form(default="default"),
):
    """Promote a query answer into the knowledge base as a synthetic raw note.

    Writes vault/{project}/raw/notes/{slug}.md, then ingests + auto-compiles it
    through the normal pipeline (shown via the live ingest stream)."""
    config = _get_config()
    project = normalize_project(project)
    try:
        rel_path = _save_note(question, answer_md, sources, project, title or question)
    except HTTPException as exc:
        return HTMLResponse(
            f'<div class="alert alert-error"><strong>Error:</strong> {escape(exc.detail)}</div>'
        )
    job_id = _spawn_ingest_job(rel_path, project, config)
    return templates.TemplateResponse(request, "ingest_stream.html", {
        "job_id": job_id,
        "target": Path(rel_path).name,
    })


@app.post("/ui/remove", response_class=HTMLResponse)
def ui_remove_fragment(
    request: Request,
    slug: str = Form(...),
    project: str = Form(default="default"),
):
    """Un-ingest the document behind an article: remove its raw source(s), the
    article, wiki points, index/citation traces. Redirects to the wiki index."""
    config = _get_config()
    project = normalize_project(project)
    try:
        validate_project(project)
    except ValueError as exc:
        return HTMLResponse(
            f'<div class="alert alert-error"><strong>Error:</strong> {escape(str(exc))}</div>'
        )
    if not _SLUG_RE.match(slug) or slug in {".", ".."}:
        return HTMLResponse('<div class="alert alert-error"><strong>Error:</strong> invalid article.</div>')

    wiki_path = f"vault/{project}/wiki/articles/{slug}.md"
    sources: list[str] = []
    db = Path(_vault_root()) / "vault" / ".search-index"
    if db.exists():
        try:
            conn = sqlite3.connect(str(db))
            sources = [r[0] for r in conn.execute(
                "SELECT raw_path FROM article_sources WHERE wiki_path = ?", (wiki_path,)
            )]
            conn.close()
        except Exception as exc:
            logger.warning("Remove: source lookup failed for %s: %s", wiki_path, exc)

    if not sources:
        return HTMLResponse(
            '<div class="alert alert-error"><strong>Nothing to remove:</strong> '
            'no source document is linked to this article (it may be an orphan — '
            'run a compile/lint to reconcile).</div>'
        )

    try:
        for raw_path in sources:
            handle_remove_doc(raw_path, _vault_root(), config, project=project)
    except LockTimeout:
        return HTMLResponse(
            '<div class="alert alert-error"><strong>Busy:</strong> the wiki is being '
            'compiled — retry in a moment.</div>'
        )
    except Exception as exc:
        logger.error("Remove failed for %s: %s", wiki_path, exc)
        return HTMLResponse(
            f'<div class="alert alert-error"><strong>Remove failed:</strong> {escape(str(exc))}</div>'
        )

    # Tell HTMX to navigate away — the article no longer exists.
    return HTMLResponse("", headers={"HX-Redirect": f"/wiki?project={project}"})


@app.post("/ui/reindex", response_class=HTMLResponse)
def ui_reindex_fragment(
    request: Request,
    project: str = Form(default="default"),
):
    """Force re-ingest every raw doc in a project and recompile the wiki fresh.
    Streamed like an ingest job (it re-embeds all docs + recompiles — slow)."""
    config = _get_config()
    project = normalize_project(project)
    try:
        validate_project(project)
    except ValueError as exc:
        return HTMLResponse(
            f'<div class="alert alert-error"><strong>Error:</strong> {escape(str(exc))}</div>'
        )
    job_id = _spawn_reindex_job(project, config)
    return templates.TemplateResponse(request, "ingest_stream.html", {
        "job_id": job_id,
        "target": f"reindex {project}",
    })


@app.post("/ui/ingest/url", response_class=HTMLResponse)
def ui_ingest_url_fragment(
    request: Request,
    url: str = Form(...),
    project: str = Form(default="default"),
):
    config = _get_config()
    project = normalize_project(project)
    try:
        validate_project(project)
    except ValueError as exc:
        return templates.TemplateResponse(request, "ingest_result.html", {
            "error": str(exc),
            "ingest_status": None, "path": "", "n_chunks": 0, "auto_compile": False,
        })
    job_id = _spawn_ingest_job(url, project, config)
    return templates.TemplateResponse(request, "ingest_stream.html", {
        "job_id": job_id,
        "target": url,
    })


@app.post("/ui/ingest/file", response_class=HTMLResponse)
def ui_ingest_file_fragment(
    request: Request,
    file: UploadFile = File(...),
    project: str = Form(default="default"),
):
    config = _get_config()
    project = normalize_project(project)
    try:
        rel_path = _save_upload(file, project)
    except HTTPException as exc:
        return templates.TemplateResponse(request, "ingest_result.html", {
            "error": exc.detail,
            "ingest_status": None, "path": "", "n_chunks": 0, "auto_compile": False,
        })

    job_id = _spawn_ingest_job(rel_path, project, config)
    return templates.TemplateResponse(request, "ingest_stream.html", {
        "job_id": job_id,
        "target": Path(rel_path).name,
    })


@app.get("/ui/ingest/active", response_class=HTMLResponse)
def ui_ingest_active(request: Request):
    """Fragment listing in-progress ingest jobs so the Ingest page can re-attach
    to their live streams after the user navigated away and came back."""
    return templates.TemplateResponse(request, "ingest_active.html", {
        "jobs": list_active(),
    })


@app.get("/ui/ingest/stream/{job_id}")
def ui_ingest_stream(job_id: str):
    """SSE stream of one ingest job's log lines; closes with the result card."""
    def event_source():
        for kind, payload in stream(job_id):
            if kind == "ping":
                yield ": keep-alive\n\n"
            elif kind == "log":
                yield f"event: log\ndata: <div>{escape(payload)}</div>\n\n"
            else:  # done — multi-line HTML payload
                data = "".join(f"data: {line}\n" for line in payload.splitlines()) or "data: \n"
                yield f"event: done\n{data}\n"
                yield "\n"
    return StreamingResponse(
        event_source(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ── dev server entry point ────────────────────────────────────────────────────

def serve(host: str | None = None, port: int | None = None, reload: bool = False) -> None:
    host = host or os.environ.get("PKMS_HOST", "0.0.0.0")
    port = port or int(os.environ.get("PKMS_PORT", "8000"))
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    uvicorn.run("pkms.web:app", host=host, port=port, reload=reload)


if __name__ == "__main__":
    serve()
