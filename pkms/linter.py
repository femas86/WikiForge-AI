import logging
import re
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pkms.db import get_stale_articles
from pkms.guards import guard_write, validate_project

logger = logging.getLogger(__name__)

# ── issue dataclasses (plain dicts for simplicity) ────────────────────────────

# Each issue: {"type": str, "severity": str, **fields}

# ── check helpers ─────────────────────────────────────────────────────────────

def _check_drift(db_path: str, project: str) -> list[dict[str, Any]]:
    issues = []
    for row in get_stale_articles(db_path, project=project):
        issues.append({
            "type": "DRIFT",
            "severity": "WARN",
            "wiki_path": row["wiki_path"],
            "raw_path": row["raw_path"],
            "stale_hash": row["stale_hash"],
            "current_hash": row["current_hash"],
        })
    return issues


def _extract_wiki_links(text: str) -> list[str]:
    """Extract [[target]] and [[target|anchor]] link targets."""
    return re.findall(r"\[\[([^\]|]+)(?:\|[^\]]*)?\]\]", text)


def _check_broken_links(vault_dir: Path, project: str) -> list[dict[str, Any]]:
    issues = []
    articles_dir = vault_dir / project / "wiki" / "articles"
    if not articles_dir.exists():
        return issues
    for article_path in sorted(articles_dir.glob("*.md")):
        try:
            text = article_path.read_text(encoding="utf-8", errors="replace")
        except Exception as exc:
            issues.append({
                "type": "READ_ERROR",
                "severity": "ERROR",
                "path": str(article_path),
                "detail": str(exc),
            })
            continue
        for target in _extract_wiki_links(text):
            # Ignore comment-inserted crosslinks that look like paths
            slug = Path(target).stem if "/" in target else target
            target_file = articles_dir / f"{slug}.md"
            if not target_file.exists():
                issues.append({
                    "type": "BROKEN_LINK",
                    "severity": "ERROR",
                    "source": str(article_path.relative_to(vault_dir.parent)),
                    "target": target,
                })
    return issues


def _check_orphans(db_path: str, vault_dir: Path, project: str) -> list[dict[str, Any]]:
    """Wiki articles whose every source has been deleted from the project's raw/."""
    issues = []
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    # Distinct wiki_paths for this project (paths embed the project segment)
    wiki_paths = [r["wiki_path"] for r in conn.execute(
        "SELECT DISTINCT wiki_path FROM article_sources WHERE wiki_path LIKE ?",
        (f"vault/{project}/%",),
    ).fetchall()]
    for wiki_path in wiki_paths:
        sources = [r["raw_path"] for r in conn.execute(
            "SELECT raw_path FROM article_sources WHERE wiki_path = ?", (wiki_path,)
        ).fetchall()]
        missing = [s for s in sources if not (vault_dir.parent / s).exists()]
        if missing and len(missing) == len(sources):
            # All sources gone → orphan
            issues.append({
                "type": "ORPHAN",
                "severity": "WARN",
                "wiki_path": wiki_path,
                "missing_sources": missing,
            })
    conn.close()
    return issues


def _check_missing_files(db_path: str, vault_dir: Path, project: str) -> list[dict[str, Any]]:
    """Files in .search-index that no longer exist on disk."""
    issues = []
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT path, collection FROM files WHERE project = ?", (project,)
    ).fetchall()
    conn.close()
    for row in rows:
        abs_path = vault_dir.parent / row["path"]
        if not abs_path.exists():
            issues.append({
                "type": "MISSING_FILE",
                "severity": "ERROR",
                "path": row["path"],
                "collection": row["collection"],
            })
    return issues


_REQUIRED_FRONTMATTER = {"title", "tags", "sources", "date"}


def _parse_frontmatter_keys(text: str) -> set[str]:
    match = re.match(r"^---\n(.*?)\n---", text, re.DOTALL)
    if not match:
        return set()
    keys = set()
    for line in match.group(1).splitlines():
        m = re.match(r"^(\w+)\s*:", line)
        if m:
            keys.add(m.group(1))
    return keys


def _check_frontmatter(vault_dir: Path, project: str) -> list[dict[str, Any]]:
    issues = []
    articles_dir = vault_dir / project / "wiki" / "articles"
    if not articles_dir.exists():
        return issues
    for article_path in sorted(articles_dir.glob("*.md")):
        try:
            text = article_path.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue  # already flagged by broken links check
        keys = _parse_frontmatter_keys(text)
        missing = _REQUIRED_FRONTMATTER - keys
        if missing:
            issues.append({
                "type": "BAD_FRONTMATTER",
                "severity": "INFO",
                "path": str(article_path.relative_to(vault_dir.parent)),
                "missing_fields": sorted(missing),
            })
    return issues


# ── report formatter ──────────────────────────────────────────────────────────

def _build_report(issues_by_type: dict[str, list], timestamp: str) -> str:
    drift = issues_by_type.get("drift", [])
    broken = issues_by_type.get("broken_links", [])
    orphans = issues_by_type.get("orphans", [])
    missing = issues_by_type.get("missing_files", [])
    frontmatter = issues_by_type.get("bad_frontmatter", [])
    total = sum(len(v) for v in issues_by_type.values())

    lines = [
        f"# PKMS Lint Report — {timestamp}",
        "",
        "## Summary",
        "| Check | Issues |",
        "|---|---|",
        f"| Drift | {len(drift)} |",
        f"| Broken links | {len(broken)} |",
        f"| Orphaned articles | {len(orphans)} |",
        f"| Missing files | {len(missing)} |",
        f"| Bad frontmatter | {len(frontmatter)} |",
        f"| **Total** | **{total}** |",
        "",
    ]

    if drift:
        lines += ["## Drift", ""]
        for i in drift:
            lines.append(f"- `{i['wiki_path']}` — source `{i['raw_path']}` changed (recompile needed)")
        lines.append("")

    if broken:
        lines += ["## Broken links", ""]
        for i in broken:
            lines.append(f"- `{i['source']}` → `[[{i['target']}]]` — article not found")
        lines.append("")

    if orphans:
        lines += ["## Orphaned articles", ""]
        for i in orphans:
            lines.append(f"- `{i['wiki_path']}` — all sources deleted: {i['missing_sources']}")
        lines.append("")

    if missing:
        lines += ["## Missing files", ""]
        for i in missing:
            lines.append(f"- `{i['path']}` ({i['collection']}) — not found on disk")
        lines.append("")

    if frontmatter:
        lines += ["## Bad frontmatter", ""]
        for i in frontmatter:
            lines.append(f"- `{i['path']}` — missing fields: {', '.join(i['missing_fields'])}")
        lines.append("")

    if total == 0:
        lines.append("_No issues found — vault is consistent._")

    return "\n".join(lines)


# ── main lint function ────────────────────────────────────────────────────────

def lint(
    vault_root: str,
    db_path: str,
    config: dict[str, Any],
    project: str = "default",
) -> dict[str, Any]:
    """Run all lint checks for a project; write vault/{project}/outputs/lint_report.md.

    Returns {issues_by_type, total_issues, report_path}.
    The Linter is read-only except for the report write.
    """
    validate_project(project)
    vault_dir = Path(vault_root) / "vault"
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    issues_by_type: dict[str, list] = {
        "drift": [],
        "broken_links": [],
        "orphans": [],
        "missing_files": [],
        "bad_frontmatter": [],
    }

    # Check 1 — Drift
    try:
        issues_by_type["drift"] = _check_drift(db_path, project)
    except Exception as exc:
        logger.error("Drift check failed: %s", exc)
        raise

    # Check 2 — Broken links
    try:
        issues_by_type["broken_links"] = _check_broken_links(vault_dir, project)
    except Exception as exc:
        logger.warning("Broken link check failed (skipping): %s", exc)

    # Check 3 — Orphaned articles
    try:
        issues_by_type["orphans"] = _check_orphans(db_path, vault_dir, project)
    except Exception as exc:
        logger.warning("Orphan check failed (skipping): %s", exc)

    # Check 4 — Missing files
    try:
        issues_by_type["missing_files"] = _check_missing_files(db_path, vault_dir, project)
    except Exception as exc:
        logger.warning("Missing file check failed (skipping): %s", exc)

    # Check 5 — Frontmatter
    try:
        issues_by_type["bad_frontmatter"] = _check_frontmatter(vault_dir, project)
    except Exception as exc:
        logger.warning("Frontmatter check failed (skipping): %s", exc)

    total_issues = sum(len(v) for v in issues_by_type.values())
    report_md = _build_report(issues_by_type, timestamp)

    # Write report
    outputs_dir = vault_dir / project / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)
    report_abs = outputs_dir / "lint_report.md"
    guard_write("linter", str(report_abs), str(vault_dir))
    try:
        report_abs.write_text(report_md, encoding="utf-8")
        report_path = f"vault/{project}/outputs/lint_report.md"
    except Exception as exc:
        logger.warning("Could not write lint report to disk: %s", exc)
        print(report_md)
        report_path = None

    logger.info("Lint done: %d issues found", total_issues)

    return {
        "issues_by_type": issues_by_type,
        "total_issues": total_issues,
        "report_path": report_path,
        "report_md": report_md,
    }
