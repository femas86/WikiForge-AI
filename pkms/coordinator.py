import argparse
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

# Load .env so CLI invocations (ingest/compile/etc.) get ANTHROPIC_API_KEY.
# Without this, only the web server (web.py) loaded it, so CLI compiles fell
# back from Claude to a slow Ollama path.
load_dotenv(Path(__file__).parent.parent / ".env")

def _build_scope(args: argparse.Namespace) -> dict[str, Any]:
    """Convert CLI compile args to the scope dict compile() expects."""
    if getattr(args, "doc", None):
        return {"type": "doc", "source": args.doc}
    if getattr(args, "topic", None):
        return {"type": "topic", "topic": args.topic}
    return {"type": "full"}

from pkms.compiler import compile as run_compile
from pkms.compiler import (
    _get_uncompiled_raw_paths,
    _git_commit,
    _git_commit_paths,
    _project_of,
    _rebuild_index,
    _strip_dangling_links,
    _wiki_path,
)
from pkms.ingest_marker import clear_ingesting, is_ingesting, mark_ingesting
from pkms.auth import ROLES, AccessDenied, role_allows
from pkms.db import (
    add_member,
    delete_article_sources,
    delete_file,
    get_article_sources,
    get_articles_for_raw,
    get_member_role,
    init_db,
    list_members,
    list_raw_paths,
    remove_member,
    upsert_file,
)
from pkms.guards import guard_write, list_projects, load_config, normalize_project, validate_project
from pkms.ingestor import fetch_and_ingest, ingest
from pkms.linter import lint as run_lint
from pkms.lock import LockTimeout, wiki_lock
from pkms.qdrant_store import delete_by_path, ensure_collections
from pkms.querier import query as run_query
from pkms.watcher import VaultWatcher

logger = logging.getLogger(__name__)

_CONFIG_PATH = Path(__file__).parent.parent / "pkms.config.yaml"
_VAULT_ROOT = Path(__file__).parent.parent
_LOCKS_DB = Path.home() / ".pkms" / "locks.db"


# ── helpers ───────────────────────────────────────────────────────────────────

def _vault_dir(vault_root: str) -> Path:
    return Path(vault_root) / "vault"


def _db_path(vault_root: str) -> str:
    return str(_vault_dir(vault_root) / ".search-index")


def _authorize(db: str, project: str, auth_user: str | None, required: str) -> None:
    """Enforce the caller's per-project role (B4).

    `auth_user` is the TRUSTED identity (reverse-proxy header) — never a
    client-supplied answer-style user. `auth_user is None` means an INTERNAL /
    system / CLI-local call (auto-compile cascade, reconcile, recover, watcher):
    authz is not enforced there — only user-facing web entry points pass a
    concrete identity. A project with NO members is treated as unclaimed/legacy
    and left open (backward compatible; guards bite once an owner adds members).
    Raises AccessDenied when a named user lacks the required role on a claimed
    project.
    """
    if auth_user is None:
        return                                       # internal / system / CLI-local
    members = list_members(db, project)
    if not members:
        return                                       # unclaimed / legacy → open
    if not role_allows(get_member_role(db, project, auth_user), required):
        raise AccessDenied(
            f"user '{auth_user}' lacks the '{required}' role on project '{project}'"
        )


def _ensure_db(vault_root: str) -> str:
    db = _db_path(vault_root)
    Path(db).parent.mkdir(parents=True, exist_ok=True)   # vault dir may not exist yet
    init_db(db)
    return db


def check_access(vault_root: str, project: str, auth_user: str | None, required: str) -> None:
    """Public authz pre-check for the web layer (raise AccessDenied before spawning
    a background job, so a denied user gets an immediate 403 rather than a failed job)."""
    _authorize(_ensure_db(vault_root), project, auth_user, required)


# ── member management (B4 (4)) ─────────────────────────────────────────────────

def _authorize_member_mgmt(
    db: str,
    project: str,
    auth_user: str | None,
    action: str,
    target_user: str | None = None,
    role: str | None = None,
) -> None:
    """Authorize a member-management action. Distinct from _authorize because of
    the bootstrap ('claim') problem: an unclaimed project has NO owner, so an
    owner-only rule would deadlock the very first assignment.

    - CLI-local / internal (auth_user is None): full access — the trusted admin
      tool. This is how the first owner is bootstrapped on a single-admin deploy.
    - list: read-only; allowed for any member of a claimed project (viewer), and
      trivially on an unclaimed one (returns []).
    - add on an UNCLAIMED project: allowed ONLY as a self-claim (caller adds
      THEMSELVES as owner). Any other add is refused ('claim it first') so a
      project can never end up claimed-but-ownerless (a second deadlock).
    - add / remove on a CLAIMED project: owner-only.
    """
    if auth_user is None:
        return
    members = list_members(db, project)
    if action == "list":
        if members and not role_allows(get_member_role(db, project, auth_user), "viewer"):
            raise AccessDenied(f"user '{auth_user}' is not a member of project '{project}'")
        return
    if not members:
        if action == "add" and target_user == auth_user and role == "owner":
            return                                       # self-claim: bootstrap
        raise AccessDenied(
            f"project '{project}' is unclaimed — claim it (add yourself as owner) "
            f"before managing members"
        )
    if not role_allows(get_member_role(db, project, auth_user), "owner"):
        raise AccessDenied(
            f"user '{auth_user}' must be an owner of project '{project}' to {action} members"
        )


def handle_member(
    action: str,
    project: str,
    vault_root: str,
    config: dict[str, Any],
    user_id: str | None = None,
    role: str | None = None,
    auth_user: str | None = None,
) -> dict[str, Any]:
    """List / add / remove project members (B4 (4)). Roles: viewer|editor|owner.

    The web layer passes the proxy-resolved identity as `auth_user`; the CLI passes
    None (trusted-local admin, full access). Refuses to remove the last owner of a
    claimed project (would strand it ownerless)."""
    validate_project(project)
    db = _ensure_db(vault_root)
    _authorize_member_mgmt(db, project, auth_user, action, target_user=user_id, role=role)

    if action == "list":
        return {"project": project, "members": list_members(db, project)}

    if action == "add":
        if role not in ROLES:
            raise ValueError(f"invalid role '{role}' — must be one of {', '.join(ROLES)}")
        if not user_id:
            raise ValueError("user_id is required to add a member")
        add_member(db, project, user_id, role, datetime.now(timezone.utc).isoformat())
        return {"project": project, "user_id": user_id, "role": role, "status": "ADDED"}

    if action == "remove":
        if not user_id:
            raise ValueError("user_id is required to remove a member")
        if get_member_role(db, project, user_id) == "owner":
            owners = [m for m in list_members(db, project) if m["role"] == "owner"]
            if len(owners) <= 1:
                raise ValueError(
                    f"cannot remove the last owner of project '{project}' "
                    f"(it would leave the project ownerless)"
                )
        remove_member(db, project, user_id)
        return {"project": project, "user_id": user_id, "status": "REMOVED"}

    raise ValueError(f"unknown member action '{action}'")


def _locks_db_path() -> str:
    path = _LOCKS_DB
    path.parent.mkdir(parents=True, exist_ok=True)
    return str(path)


def _write_ingest_result(result: dict[str, Any], vault_root: str, db_path: str) -> None:
    """Write ingest result to .search-index. Coordinator owns this write."""
    db_abs = _vault_dir(vault_root) / ".search-index"
    guard_write("coordinator", str(db_abs), str(_vault_dir(vault_root)))
    upsert_file(
        db_path,
        path=result["path"],
        hash=result["hash"],
        qdrant_ids=result["qdrant_ids"],
        collection=result.get("collection", "raw"),
        indexed_at=result["indexed_at"],
        project=result.get("project", "default"),
        content_hash=result.get("content_hash", ""),
    )
    # Audit trail (§5.1): commit the raw document to the vault's git repo, so the team
    # can verify what the vault held at any point — not only the compiled articles.
    # Deliberately AFTER the index write: the index row is the completion marker and
    # the source of truth; git is the trail. _git_commit is non-fatal (warns, never
    # raises) and no-ops when nothing changed, so a forced re-ingest of unchanged
    # bytes stays idempotent here too. Before this, raw only reached git as a side
    # effect of an un-ingest's `git add -A` (or by hand).
    project = result.get("project", "default")
    rel_path = str(Path(result["path"]).relative_to("vault"))
    _git_commit(_vault_dir(vault_root), [rel_path],
                f"ingest: {Path(result['path']).name} ({project})")


# ── verb handlers ─────────────────────────────────────────────────────────────

def handle_ingest(target: str, vault_root: str, config: dict[str, Any], project: str = "default",
                  force: bool = False, auth_user: str | None = None) -> dict[str, Any]:
    """Ingest a file path or URL into a project. Returns the ingestor result dict.

    force=True re-embeds even if the hash is unchanged (see ingest()).
    """
    validate_project(project)
    db = _ensure_db(vault_root)
    _authorize(db, project, auth_user, "editor")
    is_url = target.startswith(("http://", "https://"))
    # Mark the file being ingested so the watcher skips it for the whole
    # ingest+compile cycle (cross-process dedup). For a local file we know the
    # path up front; for a URL the destination filename isn't known until the
    # fetch derives it, so fetch_and_ingest drops the marker itself right before
    # writing the file to disk (before the watcher can see it) — here we only
    # learn the path afterwards so `finally` can clear it.
    marker = None if is_url else target
    if marker:
        mark_ingesting(marker)
    try:
        if is_url:
            result = fetch_and_ingest(target, vault_root, db, config, project=project, force=force)
            marker = result.get("path")           # fetch_and_ingest already marked this; keep it for cleanup
        else:
            result = ingest(target, vault_root, db, config, project=project, force=force)

        if result.get("status") != "SKIPPED":
            _write_ingest_result(result, vault_root, db)

        logger.info("Ingest: %s — %s", result.get("status"), result.get("path", target))

        if config["ingest"].get("auto_compile") and result.get("status") != "SKIPPED":
            logger.info("Auto-compile triggered after ingest of %s", result.get("path"))
            handle_compile(scope="all", vault_root=vault_root, config=config, project=project)

        return result
    finally:
        if marker:
            clear_ingesting(marker)


def handle_compile(scope: dict[str, Any] | str, vault_root: str, config: dict[str, Any],
                   project: str = "default", auth_user: str | None = None) -> dict[str, Any]:
    """Compile one project's wiki, holding that project's wiki lock for the duration.

    scope can be a pre-built dict or the legacy string "all" (treated as full scope).
    Lock scope is "wiki:{project}" so different projects compile in parallel.
    """
    validate_project(project)
    if isinstance(scope, str):
        if scope != "all":
            raise ValueError(f"Unknown compile scope string: {scope!r} (expected 'all' or a scope dict)")
        scope = {"type": "full"}
    _authorize(_ensure_db(vault_root), project, auth_user, "editor")  # before the lock
    locks_db = _locks_db_path()
    try:
        with wiki_lock(project, "compiler", locks_db, config) as token:
            db = _ensure_db(vault_root)
            result = run_compile(
                scope=scope,
                vault_root=vault_root,
                db_path=db,
                lock_token=token,
                config=config,
                project=project,
            )
            logger.info(
                "Compile: %s — %d articles written",
                result.get("status"),
                result.get("n_articles", 0),
            )
    except LockTimeout as exc:
        logger.error("Compile: wiki:%s lock timeout — %s", project, exc)
        raise

    # Post-compile lint runs after the lock is released — the Linter is read-only
    if config.get("compile", {}).get("post_lint") and result.get("n_articles", 0) > 0:
        logger.info("Post-compile lint triggered")
        try:
            handle_lint(vault_root=vault_root, config=config, project=project)
        except Exception as exc:
            logger.warning("Post-compile lint failed: %s", exc)
    return result


def _delete_wiki_article(
    wiki_path: str,
    vault_dir: Path,
    db_path: str,
    wiki_collection: str,
    config: dict[str, Any],
) -> None:
    """Erase one compiled article: wiki Qdrant points, files row, and the .md file."""
    delete_by_path(wiki_collection, wiki_path, config)
    delete_file(db_path, wiki_path)
    article_abs = vault_dir / Path(wiki_path).relative_to("vault")
    if article_abs.exists():
        guard_write("remover", str(article_abs), str(vault_dir))
        article_abs.unlink()


def handle_remove_doc(
    raw_path: str,
    vault_root: str,
    config: dict[str, Any],
    project: str = "default",
    auth_user: str | None = None,
) -> dict[str, Any]:
    """Un-ingest a document: remove every trace of one raw doc from a project.

    Cleans, under the project's wiki lock: the raw file + its raw Qdrant points +
    .search-index row; each article derived from it (file + wiki points + rows).
    If an article had OTHER sources, it is torn down and its remaining sources
    become uncompiled so a follow-up compile rebuilds it fresh (avoiding a
    minimal-update that would keep the removed content). Articles with no
    remaining source are deleted outright and their dangling [[links]] stripped
    from siblings. The lint report is regenerated afterwards (drops citations).

    raw_path must be the vault-relative key (vault/{project}/raw/...).
    """
    validate_project(project)
    proj = _project_of(raw_path)  # raises on non-vault-relative paths
    if proj != project:
        raise ValueError(
            f"Project mismatch for {raw_path!r}: path resolves to {proj!r} "
            f"but removal was invoked for project {project!r}."
        )
    vault_dir = _vault_dir(vault_root)
    db = _ensure_db(vault_root)
    _authorize(db, project, auth_user, "editor")
    raw_collection = config["qdrant"]["collections"]["raw"]
    wiki_collection = config["qdrant"]["collections"]["wiki"]

    locks_db = _locks_db_path()
    removed_slugs: set[str] = set()
    need_recompile = False
    raw_points = 0
    try:
        with wiki_lock(project, "remover", locks_db, config):
            # Articles compiled (partly) from this raw. Fall back to the slug-derived
            # path if article_sources has no row but the file exists on disk.
            wiki_paths = list(get_articles_for_raw(db, raw_path))
            derived = _wiki_path(raw_path, project)
            if derived not in wiki_paths and (vault_dir / Path(derived).relative_to("vault")).exists():
                wiki_paths.append(derived)

            for wiki_path in wiki_paths:
                remaining = [
                    s for s in get_article_sources(db, wiki_path) if s["raw_path"] != raw_path
                ]
                _delete_wiki_article(wiki_path, vault_dir, db, wiki_collection, config)
                delete_article_sources(db, wiki_path=wiki_path)
                if remaining:
                    need_recompile = True  # remaining sources are now uncompiled → rebuild
                else:
                    removed_slugs.add(Path(wiki_path).stem)

            # Remove the raw document itself
            raw_points = delete_by_path(raw_collection, raw_path, config)
            delete_file(db, raw_path)
            raw_abs = vault_dir / Path(raw_path).relative_to("vault")
            if raw_abs.exists():
                guard_write("remover", str(raw_abs), str(vault_dir))
                raw_abs.unlink()

            if not need_recompile:
                # Article(s) fully gone: fix siblings + prune the index now (under lock).
                # When a recompile follows, handle_compile rebuilds index/crosslinks itself.
                _strip_dangling_links(vault_dir, project, removed_slugs, config)
                try:
                    _rebuild_index(vault_dir, project)
                except Exception as exc:
                    logger.warning("Index rebuild after removal failed (non-fatal): %s", exc)

            _git_commit_paths(vault_dir, f"remove: un-ingest {Path(raw_path).name} ({project})")
    except LockTimeout as exc:
        logger.error("Remove: wiki:%s lock timeout — %s", project, exc)
        raise

    # Outside the lock (both reacquire wiki:{project} / are read-only):
    if need_recompile:
        logger.info("Remove: recompiling %s from remaining sources", project)
        handle_compile(scope="all", vault_root=vault_root, config=config, project=project)
    else:
        try:
            handle_lint(vault_root=vault_root, config=config, project=project)
        except Exception as exc:
            logger.warning("Post-removal lint failed (non-fatal): %s", exc)

    logger.info(
        "Remove: %s — %d raw points, %d article(s) deleted, recompiled=%s",
        raw_path, raw_points, len(removed_slugs), need_recompile,
    )
    return {
        "status": "REMOVED",
        "raw_path": raw_path,
        "raw_points_deleted": raw_points,
        "articles_removed": sorted(removed_slugs),
        "recompiled": need_recompile,
    }


def handle_reindex(project: str, vault_root: str, config: dict[str, Any],
                   auth_user: str | None = None) -> dict[str, Any]:
    """Rebuild a project from its raw files: force re-ingest every raw doc (backfills
    Qdrant payloads after a schema change), tear down the compiled wiki so it can't
    do a minimal-update over stale content, then recompile everything fresh.

    This is the regrounding path after the payload-text fix: unchanged raw bytes hit
    the ingest skip, so only a forced re-ingest re-embeds them with the new payload.
    """
    validate_project(project)
    db = _ensure_db(vault_root)
    _authorize(db, project, auth_user, "editor")
    raw_paths = list_raw_paths(db, project)
    if not raw_paths:
        logger.info("Reindex: no raw docs in project %s", project)
        return {"status": "SKIPPED", "project": project, "reingested": 0, "articles_rebuilt": 0}

    # 1. Force re-ingest each raw doc (re-embed + backfill payload text). No per-doc
    #    compile — we recompile once at the end.
    reingested = 0
    for rp in raw_paths:
        try:
            result = ingest(rp, vault_root, db, config, project=project, force=True)
            if result.get("status") != "SKIPPED":
                _write_ingest_result(result, vault_root, db)
                reingested += 1
        except Exception as exc:
            logger.error("Reindex: re-ingest failed for %s: %s", rp, exc)

    # 2. Tear down the compiled wiki (article file + wiki points + rows) so the
    #    follow-up compile writes fresh, grounded articles instead of minimally
    #    updating the old ungrounded ones. Under the wiki lock (mirrors remove).
    locks_db = _locks_db_path()
    wiki_collection = config["qdrant"]["collections"]["wiki"]
    torn_down = 0
    with wiki_lock(project, "remover", locks_db, config):
        seen: set[str] = set()
        for rp in raw_paths:
            for wiki_path in get_articles_for_raw(db, rp):
                if wiki_path in seen:
                    continue
                seen.add(wiki_path)
                _delete_wiki_article(wiki_path, _vault_dir(vault_root), db, wiki_collection, config)
                delete_article_sources(db, wiki_path=wiki_path)
                torn_down += 1

    # 3. Recompile everything fresh (grounded) + index + crosslink + lint. A doc
    #    that still fails (e.g. too big for any backend) is skipped, not fatal —
    #    it stays uncompiled and a later `pkms compile` retries just that one.
    compile_res = handle_compile(scope="all", vault_root=vault_root, config=config, project=project)
    compiled = compile_res.get("n_articles", 0)
    failed = compile_res.get("failed", [])

    logger.info("Reindex: %s — %d re-ingested, %d compiled, %d failed",
                project, reingested, compiled, len(failed))
    return {"status": "DONE", "project": project, "reingested": reingested,
            "articles_rebuilt": compiled, "failed": failed}


def handle_query(
    question: str,
    user_id: str,
    vault_root: str,
    config: dict[str, Any],
    session_id: str | None = None,
    prior_context: list[str] | None = None,
    project: str = "default",
    auth_user: str | None = None,
) -> dict[str, Any]:
    """Answer a question within a project. No lock needed — read-only.

    user_id is the answer-style identity (client-chosen, personalisation);
    auth_user is the TRUSTED identity used for the access check (never trust
    the client-supplied user_id for authorization)."""
    validate_project(project)
    db = _ensure_db(vault_root)
    _authorize(db, project, auth_user, "viewer")
    return run_query(
        question=question,
        user_id=user_id,
        vault_root=vault_root,
        config=config,
        session_id=session_id,
        prior_context=prior_context,
        project=project,
    )


def handle_lint(vault_root: str, config: dict[str, Any], project: str = "default",
                semantic: bool | None = None, auth_user: str | None = None) -> dict[str, Any]:
    """Run lint checks for a project. The Linter is read-only — no lock needed (see pkms_seq_lint.md).

    semantic: opt-in LLM audit (contradictions/coherence/stubs). None → config
    lint.llm_audit (default off). The auto post-compile lint leaves it None."""
    validate_project(project)
    db = _ensure_db(vault_root)
    _authorize(db, project, auth_user, "viewer")
    result = run_lint(vault_root=vault_root, db_path=db, config=config, project=project,
                      semantic=semantic)
    logger.info("Lint: %d issues — %s", result["total_issues"], result["report_path"])
    return result


def reconcile_projects(vault_root: str, config: dict[str, Any]) -> int:
    """Compile any raw left uncompiled (raw↔wiki discrepancy), per project.

    This is the watcher's startup gate: the system is "consistent" only when no
    project has uncompiled raw. Returns the number of docs that needed compiling.
    """
    db = _ensure_db(vault_root)
    vault_dir = Path(vault_root) / "vault"
    total = 0
    for project in list_projects(vault_dir):
        try:
            pending = _get_uncompiled_raw_paths(db, project)
        except Exception as exc:
            logger.warning("Reconcile: pending lookup failed for %s: %s", project, exc)
            continue
        if pending:
            logger.info("Reconcile: %d uncompiled doc(s) in project %s — compiling", len(pending), project)
            handle_compile(scope="all", vault_root=vault_root, config=config, project=project)
            total += len(pending)
    return total


def handle_export_okf(vault_root: str, config: dict[str, Any], project: str = "default",
                      out: str | None = None) -> dict[str, Any]:
    """Export a project's compiled wiki as an OKF v0.1 bundle (Open Knowledge Format,
    Google Cloud). Read-only: the canonical wiki, linter, link-graph and querier are
    untouched — OKF is a second, opt-in representation. Defaults out to <vault>/okf/<project>."""
    from pkms.okf_export import export_okf
    validate_project(project)
    out_dir = out or str(Path(vault_root) / "okf" / project)
    return export_okf(Path(vault_root) / "vault", project, out_dir)


def _watcher_ready_path() -> Path:
    return Path.home() / ".pkms" / "watcher.ready"


def handle_watch(vault_root: str, config: dict[str, Any]) -> None:
    """Reconcile raw↔wiki, signal readiness, then watch until Ctrl-C.

    The watcher is a precondition: it only reports ready (touches watcher.ready)
    after processing any raw↔wiki discrepancies, so the launch script can wait
    for it before starting the server.
    """
    db = _ensure_db(vault_root)

    # ── reconciliation gate: process discrepancies before going live ──
    try:
        n = reconcile_projects(vault_root, config)
        logger.info("Watcher: reconciliation complete — %d doc(s) compiled", n)
    except Exception as exc:
        logger.error("Watcher: reconciliation failed: %s", exc)

    ready = _watcher_ready_path()
    ready.parent.mkdir(parents=True, exist_ok=True)
    ready.write_text("ok", encoding="utf-8")
    logger.info("WATCHER READY")

    def _on_change(rel_path: str, file_hash: str) -> None:
        # Skip files the API/CLI is ingesting (cross-process dedup marker)
        if is_ingesting(rel_path):
            logger.info("Watcher: %s is being ingested via API — deferring", rel_path)
            return
        parts = Path(rel_path).parts  # vault/{project}/raw/...
        project = parts[1] if len(parts) >= 3 and parts[0] == "vault" else "default"
        logger.info("Watcher: change detected %s (project=%s)", rel_path, project)
        try:
            # pass the vault-relative path (ingest resolves the absolute internally);
            # passing an absolute path here broke the compiler's _project_of().
            result = ingest(rel_path, vault_root, db, config, project=project)
            if result.get("status") != "SKIPPED":
                _write_ingest_result(result, vault_root, db)
            if config["ingest"].get("auto_compile") and result.get("status") != "SKIPPED":
                handle_compile(scope="all", vault_root=vault_root, config=config, project=project)
        except Exception as exc:
            logger.error("Watcher ingest failed for %s: %s", rel_path, exc)

    watcher = VaultWatcher(vault_root, db, config, _on_change)
    watcher.start()
    print("Watcher running — press Ctrl-C to stop.", flush=True)
    try:
        while watcher.is_alive():
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        watcher.stop()
        ready.unlink(missing_ok=True)
        print("Watcher stopped.", flush=True)




# ── CLI entry point ───────────────────────────────────────────────────────────

def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )

    vault_root = str(_VAULT_ROOT)
    config = load_config(str(_CONFIG_PATH))

    # Ensure Qdrant collections exist before any agent runs
    try:
        ensure_collections(config)
    except Exception as exc:
        logger.warning("Qdrant not reachable at startup: %s — collections will be created on first use", exc)

    parser = argparse.ArgumentParser(
        prog="pkms",
        description="Personal Knowledge Base Management System",
    )
    sub = parser.add_subparsers(dest="verb", required=True)

    # ingest
    p_ingest = sub.add_parser("ingest", help="Ingest a file path or URL")
    p_ingest.add_argument("target", help="Absolute/relative file path or http(s):// URL")
    p_ingest.add_argument("--project", type=normalize_project, default="default", help="Project to ingest into (default: default)")
    p_ingest.add_argument("--force", action="store_true", help="Re-embed even if the file is unchanged (bypass the hash skip)")

    # reindex
    p_reindex = sub.add_parser("reindex", help="Force re-ingest all raw docs in a project and recompile the wiki fresh")
    p_reindex.add_argument("--project", type=normalize_project, default="default", help="Project to reindex (default: default)")

    # compile
    p_compile = sub.add_parser("compile", help="Compile raw documents into wiki articles")
    grp = p_compile.add_mutually_exclusive_group()
    grp.add_argument("--doc", metavar="PATH", help="Compile one document by path")
    grp.add_argument("--topic", metavar="TOPIC", help="Compile by topic (matches path or Qdrant tags)")
    p_compile.add_argument("--project", type=normalize_project, default="default", help="Project to compile (default: default)")

    # remove
    p_remove = sub.add_parser("remove", help="Un-ingest a document (remove raw + wiki + index traces)")
    p_remove.add_argument("target", help="Vault-relative raw path, e.g. vault/<project>/raw/foo.pdf")
    p_remove.add_argument("--project", type=normalize_project, default="default", help="Project the document belongs to (default: default)")

    # query
    p_query = sub.add_parser("query", help="Answer a question from the knowledge base")
    p_query.add_argument("question", help="The question to answer")
    p_query.add_argument("--user", default="default", dest="user_id", help="User ID (for Mem0)")
    p_query.add_argument("--project", type=normalize_project, default="default", help="Project to query (default: default)")

    # lint
    p_lint = sub.add_parser("lint", help="Lint the wiki for consistency issues")
    p_lint.add_argument("--project", type=normalize_project, default="default", help="Project to lint (default: default)")
    p_lint.add_argument("--semantic", action="store_true",
                        help="also run the LLM semantic audit (contradictions/coherence/stubs; costs LLM calls)")

    p_okf = sub.add_parser("export-okf",
                           help="Export the wiki as an Open Knowledge Format (OKF v0.1) bundle")
    p_okf.add_argument("--project", type=normalize_project, default="default",
                       help="Project whose wiki to export (default: default)")
    p_okf.add_argument("--out", default=None,
                       help="Output directory for the OKF bundle (default: <vault>/okf/<project>)")

    # member (B4 (4)) — project role management; CLI is the trusted-local admin path
    p_member = sub.add_parser("member", help="Manage project members (owner/editor/viewer roles)")
    msub = p_member.add_subparsers(dest="member_action", required=True)
    m_add = msub.add_parser("add", help="Add a member (or update their role)")
    m_add.add_argument("user_id", help="User identifier (matches the proxy-supplied X-Auth-User)")
    m_add.add_argument("--project", type=normalize_project, default="default", help="Project (default: default)")
    m_add.add_argument("--role", choices=list(ROLES), required=True, help="Role to grant")
    m_rm = msub.add_parser("remove", help="Remove a member")
    m_rm.add_argument("user_id", help="User identifier to remove")
    m_rm.add_argument("--project", type=normalize_project, default="default", help="Project (default: default)")
    m_ls = msub.add_parser("list", help="List members of a project")
    m_ls.add_argument("--project", type=normalize_project, default="default", help="Project (default: default)")

    # watch
    sub.add_parser("watch", help="Watch vault/raw/ for new/changed files")

    # serve
    p_serve = sub.add_parser("serve", help="Start the PKMS web API (FastAPI/uvicorn)")
    p_serve.add_argument("--host", default=None, help="Bind host (default: $PKMS_HOST or 0.0.0.0)")
    p_serve.add_argument("--port", type=int, default=None, help="Bind port (default: $PKMS_PORT or 8000)")
    p_serve.add_argument("--reload", action="store_true", help="Enable auto-reload (dev mode)")


    args = parser.parse_args()

    if args.verb == "ingest":
        result = handle_ingest(args.target, vault_root, config, project=args.project, force=args.force)
        status = result.get("status", "DONE")
        path = result.get("path", args.target)
        chunks = result.get("n_chunks", "?")
        print(f"[{status}] {path}  ({chunks} chunks)")

    elif args.verb == "reindex":
        try:
            result = handle_reindex(args.project, vault_root, config)
        except LockTimeout:
            sys.exit("Error: wiki lock timeout — retry in a moment")
        msg = (f"[{result['status']}] project {result['project']}: "
               f"{result['reingested']} re-ingested, {result['articles_rebuilt']} articles rebuilt")
        if result.get("failed"):
            msg += f", {len(result['failed'])} FAILED ({', '.join(Path(p).stem for p in result['failed'])})"
        print(msg)

    elif args.verb == "compile":
        scope = _build_scope(args)
        try:
            result = handle_compile(scope=scope, vault_root=vault_root, config=config, project=args.project)
        except LockTimeout:
            sys.exit("Error: wiki lock timeout — retry in a moment")
        written = result.get("n_articles", 0)
        status = result.get("status", "DONE")
        print(f"[{status}] {written} article(s) written")

    elif args.verb == "remove":
        try:
            result = handle_remove_doc(args.target, vault_root, config, project=args.project)
        except LockTimeout:
            sys.exit("Error: wiki lock timeout — retry in a moment")
        arts = ", ".join(result["articles_removed"]) or "—"
        print(f"[{result['status']}] {result['raw_path']}  "
              f"(raw points: {result['raw_points_deleted']}, articles removed: {arts}, "
              f"recompiled: {result['recompiled']})")

    elif args.verb == "query":
        result = handle_query(args.question, args.user_id, vault_root, config, project=args.project)
        print(result["answer_md"])
        print(f"\nCoverage: {result['coverage']}")
        if result["sources"]:
            print(f"Sources: {', '.join(result['sources'])}")

    elif args.verb == "lint":
        result = handle_lint(vault_root=vault_root, config=config, project=args.project,
                             semantic=(args.semantic or None))
        print(f"[DONE] {result['total_issues']} issue(s) — {result['report_path']}")

    elif args.verb == "export-okf":
        result = handle_export_okf(vault_root=vault_root, config=config,
                                   project=args.project, out=args.out)
        print(f"[DONE] OKF v{result['okf_version']} — {result['n_articles']} article(s) → "
              f"{result['out_dir']}")

    elif args.verb == "member":
        # CLI runs as the trusted-local admin (auth_user=None → full access).
        result = handle_member(
            args.member_action, args.project, vault_root, config,
            user_id=getattr(args, "user_id", None),
            role=getattr(args, "role", None),
            auth_user=None,
        )
        if args.member_action == "list":
            members = result["members"]
            if not members:
                print(f"project '{result['project']}': no members (unclaimed / open)")
            else:
                print(f"project '{result['project']}' members:")
                for m in members:
                    print(f"  {m['user_id']:<24} {m['role']}")
        else:
            print(f"[{result['status']}] {result['project']}: {result['user_id']}"
                  + (f" → {result['role']}" if result.get("role") else ""))

    elif args.verb == "watch":
        handle_watch(vault_root=vault_root, config=config)

    elif args.verb == "serve":
        from pkms.web import serve
        serve(host=args.host, port=args.port, reload=args.reload)



if __name__ == "__main__":
    # Fallback entry point so `python -m pkms.coordinator <verb> …` works even when
    # the console script `pkms` isn't installed (e.g. no [build-system] → uv sync
    # installs deps only). The `pkms` script, when installed, calls main() directly.
    main()
