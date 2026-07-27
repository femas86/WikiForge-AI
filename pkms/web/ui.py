"""UI routes (roadmap 1.12 split): HTML pages, HTMX fragments, and the member-
management panel. Patched collaborators (handle_query, _spawn_ingest_job, httpx,
os, _get_uncompiled_raw_paths) are reached via `web.<name>` at request time so
patch("pkms.web.<name>") still targets them; the rest is imported by name.
"""
import sqlite3
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile, status
from fastapi.responses import HTMLResponse, RedirectResponse
from markupsafe import escape

import pkms.web as web
from pkms.auth import ROLES, AccessDenied
from pkms.compiler import _extract_frontmatter_field, _extract_frontmatter_tags
from pkms.coordinator import check_access, handle_member, handle_remove_doc
from pkms.guards import normalize_project, validate_project
from pkms.lock import LockTimeout
from pkms.user_prefs import list_users, load_user_style, save_user_style
from pkms.web import (
    _FRONTMATTER_RE,
    _SLUG_RE,
    _forbidden_fragment,
    _rewrite_wikilinks,
    _save_note,
    _save_upload,
    _spawn_reindex_job,
    get_config,
    get_current_user,
    get_projects,
    get_vault_root,
    logger,
    templates,
)

router = APIRouter()


# ── UI page routes ────────────────────────────────────────────────────────────

@router.get("/", response_class=HTMLResponse)
def root():
    return RedirectResponse(url="/query")


@router.get("/query", response_class=HTMLResponse)
def ui_query_page(request: Request, projects: list[str] = Depends(get_projects)):
    cookie_user = request.cookies.get("pkms_user") or "default"
    users = sorted(set(list_users()) | {cookie_user})  # current user always selectable
    return templates.TemplateResponse(request, "query.html", {
        "active":       "query",
        "projects":     projects,
        "users":        users,
        "current_user": cookie_user,
    })


@router.get("/ingest", response_class=HTMLResponse)
def ui_ingest_page(request: Request, projects: list[str] = Depends(get_projects)):
    return templates.TemplateResponse(request, "ingest.html", {"active": "ingest", "projects": projects})


@router.get("/settings", response_class=HTMLResponse)
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


@router.get("/ui/status", response_class=HTMLResponse)
def ui_status_page(
    request: Request,
    config: dict = Depends(get_config),
    vault_root: str = Depends(get_vault_root),
    projects_list: list[str] = Depends(get_projects),
):
    vr = Path(vault_root)
    vault_dir = vr / "vault"

    # Probe Qdrant
    qdrant_ok = False
    try:
        host = config["qdrant"]["host"]
        port = config["qdrant"]["port"]
        r = web.httpx.get(f"http://{host}:{port}/healthz", timeout=2)
        qdrant_ok = r.status_code == 200
    except Exception:
        pass

    # Probe Ollama
    ollama_ok = False
    try:
        ollama_cfg = config.get("ollama", {})
        host = ollama_cfg.get("host", "localhost")
        port = ollama_cfg.get("port", 11434)
        r = web.httpx.get(f"http://{host}:{port}/api/tags", timeout=2)
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
        for p in projects_list
    ]

    return templates.TemplateResponse(request, "status.html", {
        "active":      "status",
        "vault_root":  str(vr),
        "projects":    projects,
        "db":          db.exists(),
        "qdrant_ok":   qdrant_ok,
        "ollama_ok":   ollama_ok,
        "api_key_set": bool(web.os.environ.get("ANTHROPIC_API_KEY")),
        "raw_count":   raw_count,
        "wiki_count":  wiki_count,
    })


# Link-graph domain logic lives in pkms.graph (roadmap 1.12); re-exported here
# for the existing import surface (tests, callers).
from pkms.graph import _norm_concept, _wiki_link_graph  # noqa: E402


@router.get("/wiki/graph/data")
def ui_wiki_graph_data(
    project: str = "default", concepts: bool = True, tags: bool = True,
    vault_root: str = Depends(get_vault_root),
):
    """JSON link graph for a project. `concepts`/`tags` toggle those node types."""
    project = normalize_project(project)
    validate_project(project)
    vault_dir = Path(vault_root) / "vault"
    return _wiki_link_graph(vault_dir, project, include_concepts=concepts, include_tags=tags)


@router.get("/wiki/graph", response_class=HTMLResponse)
def ui_wiki_graph_page(request: Request, project: str = "default",
                       projects: list[str] = Depends(get_projects)):
    """Force-directed graph view of a project's wiki."""
    project = normalize_project(project)
    try:
        validate_project(project)
    except ValueError:
        return RedirectResponse(url="/wiki")
    return templates.TemplateResponse(request, "wiki_graph.html", {
        "active":   "wiki",
        "project":  project,
        "projects": projects,
    })


@router.get("/wiki", response_class=HTMLResponse)
def ui_wiki_index(request: Request, project: str = "default", tag: str = "",
                  vault_root: str = Depends(get_vault_root),
                  projects: list[str] = Depends(get_projects)):
    """Browse one project's compiled wiki articles (read-only)."""
    project = normalize_project(project)
    try:
        validate_project(project)
    except ValueError:
        return RedirectResponse(url="/wiki")
    vault_dir = Path(vault_root) / "vault"
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
            pending = web._get_uncompiled_raw_paths(str(db), project)
        except Exception as exc:
            logger.warning("Pending-compile lookup failed: %s", exc)

    return templates.TemplateResponse(request, "wiki.html", {
        "active":     "wiki",
        "project":    project,
        "projects":   projects,
        "articles":   articles,
        "pending":    pending,
        "all_tags":   sorted(all_tags),
        "active_tag": active_tag,
    })


@router.get("/wiki/{project}/{slug}", response_class=HTMLResponse)
def ui_wiki_article(request: Request, project: str, slug: str,
                    vault_root: str = Depends(get_vault_root)):
    """Render one wiki article through the client-side markdown pipeline."""
    project = normalize_project(project)
    try:
        validate_project(project)
    except ValueError:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Unknown project")
    if not _SLUG_RE.match(slug) or slug in {".", ".."}:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Article not found")

    articles_dir = Path(vault_root) / "vault" / project / "wiki" / "articles"
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
    db = Path(vault_root) / "vault" / ".search-index"
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
    graph = _wiki_link_graph(Path(vault_root) / "vault", project)
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

@router.post("/ui/query", response_class=HTMLResponse)
def ui_query_fragment(
    request: Request,
    text: str = Form(...),
    user_id: str = Form(default="default"),
    project: str = Form(default="default"),
    config: dict = Depends(get_config),
    vault_root: str = Depends(get_vault_root),
    auth_user: str = Depends(get_current_user),
):
    try:
        result = web.handle_query(
            question=text,
            user_id=user_id,
            vault_root=vault_root,
            config=config,
            project=normalize_project(project),
            auth_user=auth_user,
        )
        return templates.TemplateResponse(request, "answer.html", {
            "answer_md": result["answer_md"],
            "sources":   result["sources"],
            "coverage":  result["coverage"],
            "question":  text,
            "project":   normalize_project(project),
        })
    except AccessDenied as exc:
        return _forbidden_fragment(exc)
    except Exception as exc:
        logger.error("UI query failed: %s", exc)
        return HTMLResponse(
            f'<div class="alert alert-error"><strong>Error:</strong> {escape(exc)}</div>'
        )


@router.post("/ui/settings", response_class=HTMLResponse)
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


@router.post("/ui/promote", response_class=HTMLResponse)
def ui_promote_fragment(
    request: Request,
    question: str = Form(...),
    answer_md: str = Form(...),
    sources: str = Form(default=""),
    title: str = Form(default=""),
    project: str = Form(default="default"),
    config: dict = Depends(get_config),
    vault_root: str = Depends(get_vault_root),
    auth_user: str = Depends(get_current_user),
):
    """Promote a query answer into the knowledge base as a synthetic raw note.

    Writes vault/{project}/raw/notes/{slug}.md, then ingests + auto-compiles it
    through the normal pipeline (shown via the live ingest stream)."""
    project = normalize_project(project)
    try:
        rel_path = _save_note(question, answer_md, sources, project, title or question, vault_root)
    except HTTPException as exc:
        return HTMLResponse(
            f'<div class="alert alert-error"><strong>Error:</strong> {escape(exc.detail)}</div>'
        )
    try:
        check_access(vault_root, project, auth_user, "editor")
    except AccessDenied as exc:
        return _forbidden_fragment(exc)
    job_id = web._spawn_ingest_job(rel_path, project, config, vault_root)
    return templates.TemplateResponse(request, "ingest_stream.html", {
        "job_id": job_id,
        "target": Path(rel_path).name,
    })


@router.post("/ui/remove", response_class=HTMLResponse)
def ui_remove_fragment(
    request: Request,
    slug: str = Form(...),
    project: str = Form(default="default"),
    config: dict = Depends(get_config),
    vault_root: str = Depends(get_vault_root),
    auth_user: str = Depends(get_current_user),
):
    """Un-ingest the document behind an article: remove its raw source(s), the
    article, wiki points, index/citation traces. Redirects to the wiki index."""
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
    db = Path(vault_root) / "vault" / ".search-index"
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
            handle_remove_doc(raw_path, vault_root, config, project=project,
                              auth_user=auth_user)
    except AccessDenied as exc:
        return _forbidden_fragment(exc)
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


@router.post("/ui/reindex", response_class=HTMLResponse)
def ui_reindex_fragment(
    request: Request,
    project: str = Form(default="default"),
    config: dict = Depends(get_config),
    vault_root: str = Depends(get_vault_root),
    auth_user: str = Depends(get_current_user),
):
    """Force re-ingest every raw doc in a project and recompile the wiki fresh.
    Streamed like an ingest job (it re-embeds all docs + recompiles — slow)."""
    project = normalize_project(project)
    try:
        validate_project(project)
    except ValueError as exc:
        return HTMLResponse(
            f'<div class="alert alert-error"><strong>Error:</strong> {escape(str(exc))}</div>'
        )
    try:
        check_access(vault_root, project, auth_user, "editor")
    except AccessDenied as exc:
        return _forbidden_fragment(exc)
    job_id = _spawn_reindex_job(project, config, vault_root)
    return templates.TemplateResponse(request, "ingest_stream.html", {
        "job_id": job_id,
        "target": f"reindex {project}",
    })


@router.post("/ui/ingest/url", response_class=HTMLResponse)
def ui_ingest_url_fragment(
    request: Request,
    url: str = Form(...),
    project: str = Form(default="default"),
    config: dict = Depends(get_config),
    vault_root: str = Depends(get_vault_root),
    auth_user: str = Depends(get_current_user),
):
    project = normalize_project(project)
    try:
        validate_project(project)
    except ValueError as exc:
        return templates.TemplateResponse(request, "ingest_result.html", {
            "error": str(exc),
            "ingest_status": None, "path": "", "n_chunks": 0, "auto_compile": False,
        })
    try:
        check_access(vault_root, project, auth_user, "editor")
    except AccessDenied as exc:
        return _forbidden_fragment(exc)
    job_id = web._spawn_ingest_job(url, project, config, vault_root)
    return templates.TemplateResponse(request, "ingest_stream.html", {
        "job_id": job_id,
        "target": url,
    })


@router.post("/ui/ingest/file", response_class=HTMLResponse)
def ui_ingest_file_fragment(
    request: Request,
    file: UploadFile = File(...),
    project: str = Form(default="default"),
    config: dict = Depends(get_config),
    vault_root: str = Depends(get_vault_root),
    auth_user: str = Depends(get_current_user),
):
    project = normalize_project(project)
    try:
        rel_path = _save_upload(file, project, vault_root)
    except HTTPException as exc:
        return templates.TemplateResponse(request, "ingest_result.html", {
            "error": exc.detail,
            "ingest_status": None, "path": "", "n_chunks": 0, "auto_compile": False,
        })

    try:
        check_access(vault_root, project, auth_user, "editor")
    except AccessDenied as exc:
        return _forbidden_fragment(exc)
    job_id = web._spawn_ingest_job(rel_path, project, config, vault_root)
    return templates.TemplateResponse(request, "ingest_stream.html", {
        "job_id": job_id,
        "target": Path(rel_path).name,
    })


# ── member management (B4 (4)) ──────────────────────────────────────────────────

def _members_view(
    project: str, config: dict, vault_root: str, auth_user: str,
    error: str | None = None, notice: str | None = None,
) -> dict[str, Any]:
    """Build the member-panel view model. Listing is authz-guarded: a non-member of
    a claimed project sees an access-denied panel instead of the roster."""
    try:
        members = handle_member("list", project, vault_root, config, auth_user=auth_user)["members"]
        denied = False
    except AccessDenied:
        members, denied = [], True
    my_role = next((m["role"] for m in members if m["user_id"] == auth_user), None)
    return {
        "project":    project,
        "members":    members,
        "auth_user":  auth_user,
        "my_role":    my_role,
        "denied":     denied,
        "claimed":    denied or bool(members),
        "can_manage": my_role == "owner",
        "can_claim":  (not members) and (not denied),
        "roles":      list(ROLES),
        "error":      error,
        "notice":     notice,
    }


def _members_fragment(
    request: Request, project: str, config: dict, vault_root: str, auth_user: str,
    error: str | None = None, notice: str | None = None,
) -> HTMLResponse:
    view = _members_view(project, config, vault_root, auth_user, error=error, notice=notice)
    return templates.TemplateResponse(request, "members_panel.html", view)


@router.get("/members", response_class=HTMLResponse)
def ui_members_page(
    request: Request,
    project: str = "default",
    config: dict = Depends(get_config),
    vault_root: str = Depends(get_vault_root),
    projects: list[str] = Depends(get_projects),
    auth_user: str = Depends(get_current_user),
):
    """Owner-managed project membership (B4). The panel adapts to the caller's
    role: owners get add/remove controls, an unclaimed project offers a claim
    button, everyone else sees a read-only roster (or an access notice)."""
    project = normalize_project(project)
    try:
        validate_project(project)
    except ValueError:
        return RedirectResponse(url="/members")
    view = _members_view(project, config, vault_root, auth_user)
    view.update({"active": "members", "projects": projects})
    return templates.TemplateResponse(request, "members.html", view)


@router.post("/ui/members/add", response_class=HTMLResponse)
def ui_members_add(
    request: Request,
    project: str = Form(default="default"),
    user_id: str = Form(...),
    role: str = Form(...),
    config: dict = Depends(get_config),
    vault_root: str = Depends(get_vault_root),
    auth_user: str = Depends(get_current_user),
):
    project = normalize_project(project)
    try:
        handle_member("add", project, vault_root, config,
                      user_id=user_id.strip(), role=role, auth_user=auth_user)
    except AccessDenied as exc:
        return _forbidden_fragment(exc)
    except ValueError as exc:
        return _members_fragment(request, project, config, vault_root, auth_user, error=str(exc))
    return _members_fragment(request, project, config, vault_root, auth_user,
                             notice=f"{user_id.strip()} → {role}")


@router.post("/ui/members/remove", response_class=HTMLResponse)
def ui_members_remove(
    request: Request,
    project: str = Form(default="default"),
    user_id: str = Form(...),
    config: dict = Depends(get_config),
    vault_root: str = Depends(get_vault_root),
    auth_user: str = Depends(get_current_user),
):
    project = normalize_project(project)
    try:
        handle_member("remove", project, vault_root, config,
                      user_id=user_id.strip(), auth_user=auth_user)
    except AccessDenied as exc:
        return _forbidden_fragment(exc)
    except ValueError as exc:
        return _members_fragment(request, project, config, vault_root, auth_user, error=str(exc))
    return _members_fragment(request, project, config, vault_root, auth_user,
                             notice=f"removed {user_id.strip()}")


