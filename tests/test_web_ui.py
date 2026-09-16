import io
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

import pkms.web as web_module
from pkms.web import app, get_config, get_vault_root
from tests._webdi import _override

CLIENT = TestClient(app, raise_server_exceptions=False)

CFG = {
    "watcher": {"enabled": True, "debounce_seconds": 0.05, "watch_paths": ["raw/"]},
    "ingest": {"supported_extensions": [".md", ".pdf", ".html", ".txt"], "auto_compile": False},
    "qdrant": {"host": "localhost", "port": 6333,
               "collections": {"raw": "raw", "wiki": "wiki", "outputs": "outputs"}},
    "embedding": {"dimension": 4},
    "chunking": {"max_tokens": 200},
    "compile": {"crosslink_threshold": 0.82, "post_lint": False},
    "query": {"top_k_wiki": 3, "top_k_raw": 3, "mem0_recall": 2},
    "lock": {"ttl_seconds": 300, "max_wait_seconds": 1},
    "ollama": {"host": "localhost", "port": 11434},
}

QUERY_RESULT = {
    "answer_md": "**Transformers** use self-attention mechanisms.",
    "sources": ["vault/wiki/articles/transformers.md"],
    "coverage": "full",
    "session_id": "s1",
    "output_path": "vault/outputs/s1.md",
}

INGEST_RESULT = {
    "status": "DONE",
    "path": "vault/raw/paper.pdf",
    "hash": "sha256:abc",
    "qdrant_ids": ["id1"],
    "collection": "raw",
    "n_chunks": 5,
    "indexed_at": "2026-06-11T10:00:00Z",
}


# ── page routes ───────────────────────────────────────────────────────────────

def test_root_redirects_to_query():
    resp = CLIENT.get("/", follow_redirects=False)
    assert resp.status_code in (301, 302, 307)
    assert resp.headers["location"] == "/query"


def test_query_page_returns_html():
    resp = CLIENT.get("/query")
    assert resp.status_code == 200
    assert "text/html" in resp.headers["content-type"]
    assert "Ask the Knowledge Base" in resp.text
    assert 'hx-post="/ui/query"' in resp.text
    assert 'href="/wiki"' in resp.text     # nav link to the wiki


def test_ingest_page_returns_html():
    resp = CLIENT.get("/ingest")
    assert resp.status_code == 200
    assert "Add to the Knowledge Base" in resp.text
    assert 'hx-post="/ui/ingest/url"' in resp.text
    assert 'hx-post="/ui/ingest/file"' in resp.text


def test_status_page_returns_html(tmp_path):
    with _override(get_vault_root, lambda: str(tmp_path)), \
         _override(get_config, lambda: CFG), \
         patch("pkms.web.httpx.get", side_effect=ConnectionError):
        resp = CLIENT.get("/ui/status")
    assert resp.status_code == 200
    assert "System Status" in resp.text
    assert "Qdrant" in resp.text
    assert "Ollama" in resp.text


def test_status_page_shows_api_key_missing(tmp_path):
    with _override(get_vault_root, lambda: str(tmp_path)), \
         _override(get_config, lambda: CFG), \
         patch("pkms.web.httpx.get", side_effect=ConnectionError), \
         patch("pkms.web.os.environ.get", return_value=None):
        resp = CLIENT.get("/ui/status")
    assert "not set" in resp.text


def test_status_page_shows_api_key_set(tmp_path):
    with _override(get_vault_root, lambda: str(tmp_path)), \
         _override(get_config, lambda: CFG), \
         patch("pkms.web.httpx.get", side_effect=ConnectionError), \
         patch("pkms.web.os.environ.get", return_value="sk-ant-test"):
        resp = CLIENT.get("/ui/status")
    assert "✓ set" in resp.text


# ── HTMX fragment: /ui/query ──────────────────────────────────────────────────

def test_ui_query_returns_answer_html(tmp_path):
    with _override(get_vault_root, lambda: str(tmp_path)), \
         _override(get_config, lambda: CFG), \
         patch("pkms.web.handle_query", return_value=QUERY_RESULT):
        resp = CLIENT.post("/ui/query", data={"text": "What are transformers?", "user_id": "alice"})
    assert resp.status_code == 200
    assert "answer-card" in resp.text
    assert "full" in resp.text            # coverage badge
    assert "data-markdown" in resp.text   # client-side markdown-render hook
    assert "transformers.md" in resp.text  # sources rendered in the fragment


def test_ui_query_shows_raw_only_warning(tmp_path):
    result = {**QUERY_RESULT, "coverage": "raw_only"}
    with _override(get_vault_root, lambda: str(tmp_path)), \
         _override(get_config, lambda: CFG), \
         patch("pkms.web.handle_query", return_value=result):
        resp = CLIENT.post("/ui/query", data={"text": "Q?", "user_id": "alice"})
    assert "pkms compile" in resp.text


def test_ui_query_error_returns_alert(tmp_path):
    with _override(get_vault_root, lambda: str(tmp_path)), \
         _override(get_config, lambda: CFG), \
         patch("pkms.web.handle_query", side_effect=RuntimeError("embed failed")):
        resp = CLIENT.post("/ui/query", data={"text": "Q?", "user_id": "alice"})
    assert resp.status_code == 200  # HTMX always 200
    assert "alert-error" in resp.text
    assert "embed failed" in resp.text


def test_ui_query_error_escapes_html(tmp_path):
    with _override(get_vault_root, lambda: str(tmp_path)), \
         _override(get_config, lambda: CFG), \
         patch("pkms.web.handle_query", side_effect=RuntimeError('<script>alert(1)</script>')):
        resp = CLIENT.post("/ui/query", data={"text": "Q?", "user_id": "alice"})
    assert "<script>alert(1)</script>" not in resp.text
    assert "&lt;script&gt;" in resp.text


def test_base_page_loads_dompurify():
    resp = CLIENT.get("/query")
    assert "dompurify" in resp.text.lower()
    assert "DOMPurify.sanitize" in resp.text


# ── SSE helpers ──────────────────────────────────────────────────────────────

def _extract_job_id(html: str) -> str:
    import re
    m = re.search(r'sse-connect="/ui/ingest/stream/([a-f0-9]+)"', html)
    assert m, f"no sse-connect fragment in response: {html[:200]}"
    return m.group(1)


def _consume_stream(job_id: str) -> str:
    """Read the SSE stream until the done event; return the full raw text."""
    with CLIENT.stream("GET", f"/ui/ingest/stream/{job_id}") as resp:
        assert resp.status_code == 200
        assert "text/event-stream" in resp.headers["content-type"]
        chunks = []
        for line in resp.iter_lines():
            chunks.append(line)
        return "\n".join(chunks)


# ── HTMX fragment: /ui/ingest/url ────────────────────────────────────────────

def test_ui_ingest_url_returns_stream_fragment(tmp_path):
    with _override(get_vault_root, lambda: str(tmp_path)), \
         _override(get_config, lambda: CFG), \
         patch("pkms.web.handle_ingest", return_value=INGEST_RESULT):
        resp = CLIENT.post("/ui/ingest/url", data={"url": "https://arxiv.org/abs/1706.03762"})
        assert resp.status_code == 200
        assert 'hx-ext="sse"' in resp.text
        job_id = _extract_job_id(resp.text)
        raw = _consume_stream(job_id)
    assert "event: done" in raw
    assert "alert-success" in raw
    assert "5 chunk" in raw


def test_ui_ingest_url_skipped(tmp_path):
    skipped = {**INGEST_RESULT, "status": "SKIPPED"}
    with _override(get_vault_root, lambda: str(tmp_path)), \
         _override(get_config, lambda: CFG), \
         patch("pkms.web.handle_ingest", return_value=skipped):
        resp = CLIENT.post("/ui/ingest/url", data={"url": "https://arxiv.org/abs/1706.03762"})
        raw = _consume_stream(_extract_job_id(resp.text))
    assert "alert-warn" in raw
    assert "Skipped" in raw


def test_ui_ingest_url_error_streams_error_card(tmp_path):
    with _override(get_vault_root, lambda: str(tmp_path)), \
         _override(get_config, lambda: CFG), \
         patch("pkms.web.handle_ingest", side_effect=ConnectionError("fetch failed")):
        resp = CLIENT.post("/ui/ingest/url", data={"url": "https://example.com/paper.pdf"})
        raw = _consume_stream(_extract_job_id(resp.text))
    assert "alert-error" in raw
    assert "fetch failed" in raw


def test_ui_ingest_url_invalid_project_immediate_error(tmp_path):
    with _override(get_vault_root, lambda: str(tmp_path)), \
         _override(get_config, lambda: CFG):
        resp = CLIENT.post("/ui/ingest/url",
                           data={"url": "https://x.org/p.pdf", "project": "../bad"})
    assert "alert-error" in resp.text  # no job spawned
    assert 'hx-ext="sse"' not in resp.text


# ── HTMX fragment: /ui/ingest/file ───────────────────────────────────────────

def test_ui_ingest_file_success(tmp_path):
    (tmp_path / "vault" / "default" / "raw").mkdir(parents=True)
    with _override(get_vault_root, lambda: str(tmp_path)), \
         _override(get_config, lambda: CFG), \
         patch("pkms.web.handle_ingest", return_value=INGEST_RESULT):
        resp = CLIENT.post(
            "/ui/ingest/file",
            files={"file": ("paper.pdf", io.BytesIO(b"content"), "application/pdf")},
        )
        assert resp.status_code == 200
        raw = _consume_stream(_extract_job_id(resp.text))
    assert "alert-success" in raw


def test_ui_ingest_file_traversal_stays_in_raw(tmp_path):
    (tmp_path / "vault" / "default" / "raw").mkdir(parents=True)
    (tmp_path / "vault" / "default" / "wiki").mkdir(parents=True)
    with _override(get_vault_root, lambda: str(tmp_path)), \
         _override(get_config, lambda: CFG), \
         patch("pkms.web.handle_ingest", return_value=INGEST_RESULT):
        resp = CLIENT.post(
            "/ui/ingest/file",
            files={"file": ("../../wiki/pwn.md", io.BytesIO(b"x"), "text/markdown")},
        )
    assert resp.status_code == 200
    assert not (tmp_path / "vault" / "default" / "wiki" / "pwn.md").exists()
    assert (tmp_path / "vault" / "default" / "raw" / "pwn.md").exists()


def test_ingest_file_endpoint_traversal_stays_in_raw(tmp_path):
    (tmp_path / "vault" / "default" / "raw").mkdir(parents=True)
    with _override(get_vault_root, lambda: str(tmp_path)), \
         _override(get_config, lambda: CFG), \
         patch("pkms.web.handle_ingest", return_value=INGEST_RESULT):
        resp = CLIENT.post(
            "/ingest/file",
            files={"file": ("../../../outside.txt", io.BytesIO(b"x"), "text/plain")},
        )
    assert resp.status_code == 200
    assert (tmp_path / "vault" / "default" / "raw" / "outside.txt").exists()
    assert not (tmp_path / "outside.txt").exists()


def test_ui_ingest_file_error_streams_error_card(tmp_path):
    (tmp_path / "vault" / "default" / "raw").mkdir(parents=True)
    with _override(get_vault_root, lambda: str(tmp_path)), \
         _override(get_config, lambda: CFG), \
         patch("pkms.web.handle_ingest", side_effect=RuntimeError("parse error")):
        resp = CLIENT.post(
            "/ui/ingest/file",
            files={"file": ("doc.pdf", io.BytesIO(b"x"), "application/pdf")},
        )
        assert resp.status_code == 200
        raw = _consume_stream(_extract_job_id(resp.text))
    assert "alert-error" in raw
    assert "parse error" in raw


# ── project routing ──────────────────────────────────────────────────────────

def test_ingest_file_endpoint_creates_new_project(tmp_path):
    with _override(get_vault_root, lambda: str(tmp_path)), \
         _override(get_config, lambda: CFG), \
         patch("pkms.web.handle_ingest", return_value=INGEST_RESULT) as mock_i:
        resp = CLIENT.post(
            "/ingest/file",
            data={"project": "robotics"},
            files={"file": ("doc.pdf", io.BytesIO(b"x"), "application/pdf")},
        )
    assert resp.status_code == 200
    assert (tmp_path / "vault" / "robotics" / "raw" / "doc.pdf").exists()
    assert mock_i.call_args.kwargs["project"] == "robotics"


def test_ingest_file_endpoint_rejects_invalid_project(tmp_path):
    with _override(get_vault_root, lambda: str(tmp_path)), \
         _override(get_config, lambda: CFG):
        resp = CLIENT.post(
            "/ingest/file",
            data={"project": "../escape"},
            files={"file": ("doc.pdf", io.BytesIO(b"x"), "application/pdf")},
        )
    assert resp.status_code == 400
    assert not (tmp_path / "escape").exists()


# ── wiki browser ──────────────────────────────────────────────────────────────

ARTICLE_MD = """---
title: Transformer Architecture
tags: [attention, nlp]
sources: [vault/default/raw/attention.pdf]
date: 2026-06-12
summary_1line: Self-attention based sequence model.
---

Transformers rely on self-attention. See [[bert]] and [[gpt|GPT models]].
"""


def _make_wiki(tmp_path, project="default"):
    articles = tmp_path / "vault" / project / "wiki" / "articles"
    articles.mkdir(parents=True)
    (articles / "transformers.md").write_text(ARTICLE_MD)
    return articles


def test_wiki_index_lists_articles(tmp_path):
    _make_wiki(tmp_path)
    with _override(get_vault_root, lambda: str(tmp_path)):
        resp = CLIENT.get("/wiki")
    assert resp.status_code == 200
    assert "Transformer Architecture" in resp.text
    assert "Self-attention based sequence model." in resp.text
    assert '/wiki/default/transformers' in resp.text


def test_wiki_index_empty_project(tmp_path):
    (tmp_path / "vault").mkdir()
    with _override(get_vault_root, lambda: str(tmp_path)):
        resp = CLIENT.get("/wiki")
    assert resp.status_code == 200
    assert "No compiled articles" in resp.text


def test_wiki_index_shows_pending_compile(tmp_path):
    from pkms.db import init_db, upsert_file
    (tmp_path / "vault").mkdir()
    db = str(tmp_path / "vault" / ".search-index")
    init_db(db)
    upsert_file(db, path="vault/default/raw/new.pdf", hash="sha256:x",
                qdrant_ids=["i"], collection="raw", indexed_at="t", project="default")
    with _override(get_vault_root, lambda: str(tmp_path)):
        resp = CLIENT.get("/wiki")
    assert "Not yet compiled" in resp.text
    assert "vault/default/raw/new.pdf" in resp.text


def test_wiki_article_renders_body_and_meta(tmp_path):
    _make_wiki(tmp_path)
    with _override(get_vault_root, lambda: str(tmp_path)):
        resp = CLIENT.get("/wiki/default/transformers")
    assert resp.status_code == 200
    assert "Transformer Architecture" in resp.text
    assert "data-markdown" in resp.text
    # frontmatter stripped from the rendered body
    assert "summary_1line" not in resp.text
    # wikilinks rewritten to browser routes
    assert "(/wiki/default/bert)" in resp.text
    assert "[GPT models](/wiki/default/gpt)" in resp.text


def test_wiki_article_shows_sources_from_db(tmp_path):
    from pkms.db import init_db, upsert_article_source
    _make_wiki(tmp_path)
    db = str(tmp_path / "vault" / ".search-index")
    init_db(db)
    upsert_article_source(db, wiki_path="vault/default/wiki/articles/transformers.md",
                          raw_path="vault/default/raw/attention.pdf",
                          compile_hash="sha256:a", compiled_at="t")
    with _override(get_vault_root, lambda: str(tmp_path)):
        resp = CLIENT.get("/wiki/default/transformers")
    assert "vault/default/raw/attention.pdf" in resp.text


def test_wiki_article_missing_returns_404(tmp_path):
    _make_wiki(tmp_path)
    with _override(get_vault_root, lambda: str(tmp_path)):
        resp = CLIENT.get("/wiki/default/nonexistent")
    assert resp.status_code == 404


def test_wiki_article_slug_traversal_blocked(tmp_path):
    _make_wiki(tmp_path)
    (tmp_path / "vault" / "default" / "wiki" / "secret.md").write_text("hidden")
    with _override(get_vault_root, lambda: str(tmp_path)):
        # encoded traversal reaches the route and must 404
        for slug in ("..%2Fsecret", ".%2E"):
            resp = CLIENT.get(f"/wiki/default/{slug}")
            assert resp.status_code == 404, slug
        # plain ".." is normalised away by the client; either way: no leak
        resp = CLIENT.get("/wiki/default/..")
        assert "hidden" not in resp.text


def test_wiki_article_invalid_project_404(tmp_path):
    with _override(get_vault_root, lambda: str(tmp_path)):
        resp = CLIENT.get("/wiki/UPPER/anything")
    assert resp.status_code == 404


def test_ingest_file_normalizes_project_case(tmp_path):
    with _override(get_vault_root, lambda: str(tmp_path)), \
         _override(get_config, lambda: CFG), \
         patch("pkms.web.handle_ingest", return_value=INGEST_RESULT) as mock_i:
        resp = CLIENT.post(
            "/ingest/file",
            data={"project": "OG-MDAI"},
            files={"file": ("doc.pdf", io.BytesIO(b"x"), "application/pdf")},
        )
    assert resp.status_code == 200
    assert (tmp_path / "vault" / "og-mdai" / "raw" / "doc.pdf").exists()
    assert mock_i.call_args.kwargs["project"] == "og-mdai"


# ── ingest job re-attach ──────────────────────────────────────────────────────

def test_ingest_active_lists_running_jobs():
    import pkms.events as events
    events._jobs.clear()
    job = events.create_job(target="paper.pdf → demo")
    try:
        resp = CLIENT.get("/ui/ingest/active")
        assert resp.status_code == 200
        assert "In progress" in resp.text
        assert "paper.pdf → demo" in resp.text
        # re-attaches via the SSE stream for that job
        assert f"/ui/ingest/stream/{job}" in resp.text
    finally:
        events._jobs.clear()


def test_ingest_active_empty_when_no_jobs():
    import pkms.events as events
    events._jobs.clear()
    resp = CLIENT.get("/ui/ingest/active")
    assert resp.status_code == 200
    assert "In progress" not in resp.text


# ── settings (per-user answer style) ──────────────────────────────────────────

def test_settings_page_renders():
    resp = CLIENT.get("/settings?user_id=alice")
    assert resp.status_code == 200
    assert "Answer style" in resp.text
    assert "alice" in resp.text


def test_settings_save_persists(tmp_path):
    import pkms.user_prefs as up
    with patch.object(up, "_users_dir", lambda: tmp_path / "users"):
        resp = CLIENT.post("/ui/settings", data={
            "user_id": "alice",
            "style": "Answer in Italian. TL;DR first.",
        })
        assert resp.status_code == 200
        assert "Saved answer style" in resp.text
        assert up.load_user_style("alice") == "Answer in Italian. TL;DR first."


def test_settings_remembers_user_via_cookie(tmp_path):
    import pkms.user_prefs as up
    from fastapi.testclient import TestClient as _TC
    with patch.object(up, "_users_dir", lambda: tmp_path / "users"):
        c = _TC(app)  # isolated cookie jar
        r = c.post("/ui/settings", data={"user_id": "federico", "style": "Rispondi in italiano."})
        assert r.status_code == 200
        assert c.cookies.get("pkms_user") == "federico"
        # Re-enter via the nav link (no query string) — must load federico's style, not "default"
        g = c.get("/settings")
        assert g.status_code == 200
        assert "Rispondi in italiano." in g.text
        assert "federico" in g.text


def test_settings_query_param_overrides_cookie(tmp_path):
    import pkms.user_prefs as up
    from fastapi.testclient import TestClient as _TC
    with patch.object(up, "_users_dir", lambda: tmp_path / "users"):
        up.save_user_style("alice", "Alice style here")
        c = _TC(app)
        c.cookies.set("pkms_user", "federico")
        g = c.get("/settings?user_id=alice")
        assert "Alice style here" in g.text


def test_query_page_user_is_a_select(tmp_path):
    import pkms.user_prefs as up
    with patch.object(up, "_users_dir", lambda: tmp_path / "users"):
        up.save_user_style("alice", "x")
        resp = CLIENT.get("/query")
    assert resp.status_code == 200
    assert '<select id="user_id" name="user_id">' in resp.text
    assert "<option value=\"alice\"" in resp.text
    assert "<option value=\"default\"" in resp.text


def test_query_page_preselects_cookie_user(tmp_path):
    import pkms.user_prefs as up
    from fastapi.testclient import TestClient as _TC
    with patch.object(up, "_users_dir", lambda: tmp_path / "users"):
        up.save_user_style("alice", "x")
        c = _TC(app)
        c.cookies.set("pkms_user", "alice")
        resp = c.get("/query")
    assert '<option value="alice" selected>' in resp.text


def test_settings_page_user_select_has_new_option(tmp_path):
    import pkms.user_prefs as up
    with patch.object(up, "_users_dir", lambda: tmp_path / "users"):
        up.save_user_style("alice", "x")
        resp = CLIENT.get("/settings?user_id=alice")
    assert resp.status_code == 200
    assert '<select id="user-select" name="user_id"' in resp.text
    assert "➕ New user…" in resp.text


# ── "Trasforma in nota" (promote answer → raw/) ───────────────────────────────

def test_promote_writes_note_and_ingests(tmp_path):
    with _override(get_vault_root, lambda: str(tmp_path)), \
         _override(get_config, lambda: CFG), \
         patch("pkms.web._spawn_ingest_job", return_value="job-xyz") as mock_job:
        resp = CLIENT.post("/ui/promote", data={
            "question": "What is attention?",
            "answer_md": "Attention weighs token relevance.",
            "sources": "[[transformers]] · raw:p.pdf#0",
            "title": "Attention explained",
            "project": "og-mdai",
        })
    assert resp.status_code == 200
    note = tmp_path / "vault" / "og-mdai" / "raw" / "notes" / "attention_explained.md"
    assert note.exists()
    text = note.read_text()
    assert "**Domanda:** What is attention?" in text
    assert "Attention weighs token relevance." in text
    # ingest+compile triggered with the vault-relative note path + project
    assert mock_job.call_args.args[0] == "vault/og-mdai/raw/notes/attention_explained.md"
    assert mock_job.call_args.args[1] == "og-mdai"


def test_promote_rejects_invalid_project(tmp_path):
    with _override(get_vault_root, lambda: str(tmp_path)), \
         _override(get_config, lambda: CFG), \
         patch("pkms.web._spawn_ingest_job") as mock_job:
        resp = CLIENT.post("/ui/promote", data={
            "question": "q", "answer_md": "a", "sources": "", "title": "t",
            "project": "../escape",
        })
    assert resp.status_code == 200
    assert "Error" in resp.text
    mock_job.assert_not_called()
    assert not (tmp_path / "escape").exists()


# ── wiki graph + backlinks + tag filter ───────────────────────────────────────

_GRAPH_A = "---\ntitle: Alpha\ntags: [core]\n---\nSee [[beta]] and [[gamma]] and [[ghost]].\n"
_GRAPH_B = "---\ntitle: Beta\ntags: [core, extra]\n---\nBack to [[alpha]].\n"
_GRAPH_C = "---\ntitle: Gamma\ntags: [extra]\n---\nNo links here.\n"


def _make_graph_wiki(tmp_path, project="default"):
    articles = tmp_path / "vault" / project / "wiki" / "articles"
    articles.mkdir(parents=True)
    (articles / "alpha.md").write_text(_GRAPH_A)
    (articles / "beta.md").write_text(_GRAPH_B)
    (articles / "gamma.md").write_text(_GRAPH_C)
    return articles


def test_wiki_graph_data(tmp_path):
    _make_graph_wiki(tmp_path)
    with _override(get_vault_root, lambda: str(tmp_path)):
        resp = CLIENT.get("/wiki/graph/data?project=default")
    assert resp.status_code == 200
    data = resp.json()
    by_type = {}
    for n in data["nodes"]:
        by_type.setdefault(n["type"], set()).add(n["id"])
    # articles
    assert by_type["article"] == {"alpha", "beta", "gamma"}
    # the unresolved [[ghost]] link is now a concept node (Obsidian-style), not dropped
    assert "concept:ghost" in by_type["concept"]
    # tags become nodes too
    assert "tag:core" in by_type.get("tag", set())
    assert "tag:extra" in by_type.get("tag", set())
    pairs = {(l["source"], l["target"]) for l in data["links"]}
    assert ("alpha", "beta") in pairs          # article → article (resolved)
    assert ("beta", "alpha") in pairs
    assert ("alpha", "concept:ghost") in pairs  # article → concept (unresolved)
    assert ("alpha", "tag:core") in pairs       # article → tag


def test_wiki_link_graph_caches_until_articles_change(tmp_path):
    import os
    from pkms.web import _wiki_link_graph
    _make_graph_wiki(tmp_path)
    articles = tmp_path / "vault" / "default" / "wiki" / "articles"
    target = next(iter(sorted(articles.glob("*.md"))))

    g1 = _wiki_link_graph(tmp_path / "vault", "default")
    with patch("pkms.graph._build_wiki_link_graph") as mock_build:
        g2 = _wiki_link_graph(tmp_path / "vault", "default")
    mock_build.assert_not_called()   # unchanged articles → no re-read of the wiki
    assert g2 is g1                  # same shared (read-only) object

    # touching one article invalidates the stamp → rebuild
    st = target.stat()
    os.utime(target, ns=(st.st_atime_ns, st.st_mtime_ns + 1_000_000))
    g3 = _wiki_link_graph(tmp_path / "vault", "default")
    assert g3 is not g1
    assert {n["id"] for n in g3["nodes"]} == {n["id"] for n in g1["nodes"]}


def test_wiki_graph_page_renders(tmp_path):
    _make_graph_wiki(tmp_path)
    with _override(get_vault_root, lambda: str(tmp_path)):
        resp = CLIENT.get("/wiki/graph?project=default")
    assert resp.status_code == 200
    assert "Wiki graph" in resp.text
    assert "force-graph" in resp.text


def test_article_shows_backlinks(tmp_path):
    _make_graph_wiki(tmp_path)
    with _override(get_vault_root, lambda: str(tmp_path)):
        resp = CLIENT.get("/wiki/default/alpha")
    assert resp.status_code == 200
    assert "Linked from" in resp.text
    assert "Beta" in resp.text
    assert "/wiki/default/beta" in resp.text


def test_wiki_tag_filter(tmp_path):
    _make_graph_wiki(tmp_path)
    with _override(get_vault_root, lambda: str(tmp_path)):
        resp = CLIENT.get("/wiki?project=default&tag=extra")
    assert resp.status_code == 200
    # beta + gamma carry 'extra'; alpha does not → its title is filtered out
    assert "Beta" in resp.text
    assert "Gamma" in resp.text
    assert "Alpha" not in resp.text


# ── startup recovery of orphaned (uncompiled) raw ─────────────────────────────

def test_recover_uncompiled_compiles_only_pending_projects(tmp_path):
    db = tmp_path / "vault" / ".search-index"
    db.parent.mkdir(parents=True)
    db.write_text("")  # must merely exist
    with patch.object(web_module, "_vault_root", return_value=str(tmp_path)), \
         patch("pkms.web._watcher_ready_path", return_value=tmp_path / "no-watcher.ready"), \
         patch.object(web_module, "_projects", return_value=["demo", "empty"]), \
         patch("pkms.web._get_uncompiled_raw_paths",
               side_effect=lambda d, p: ["vault/demo/raw/x.pdf"] if p == "demo" else []), \
         patch("pkms.web.handle_compile") as mock_compile:
        web_module._recover_uncompiled(CFG)
    assert mock_compile.call_count == 1
    assert mock_compile.call_args.kwargs["project"] == "demo"


def test_recover_uncompiled_skips_when_watcher_running(tmp_path):
    # watcher.ready present → the watcher owns reconciliation → web recovery must skip
    db = tmp_path / "vault" / ".search-index"
    db.parent.mkdir(parents=True)
    db.write_text("")
    ready = tmp_path / "watcher.ready"
    ready.write_text("ok")
    with patch.object(web_module, "_vault_root", return_value=str(tmp_path)), \
         patch("pkms.web._watcher_ready_path", return_value=ready), \
         patch.object(web_module, "_projects", return_value=["demo"]), \
         patch("pkms.web._get_uncompiled_raw_paths", return_value=["vault/demo/raw/x.pdf"]), \
         patch("pkms.web.handle_compile") as mock_compile:
        web_module._recover_uncompiled(CFG)
    mock_compile.assert_not_called()


def test_recover_uncompiled_nonfatal_on_compile_error(tmp_path):
    db = tmp_path / "vault" / ".search-index"
    db.parent.mkdir(parents=True)
    db.write_text("")
    with patch.object(web_module, "_vault_root", return_value=str(tmp_path)), \
         patch("pkms.web._watcher_ready_path", return_value=tmp_path / "no-watcher.ready"), \
         patch.object(web_module, "_projects", return_value=["demo"]), \
         patch("pkms.web._get_uncompiled_raw_paths", return_value=["vault/demo/raw/x.pdf"]), \
         patch("pkms.web.handle_compile", side_effect=RuntimeError("compile boom")):
        web_module._recover_uncompiled(CFG)  # must not raise


def test_recover_uncompiled_noop_without_db(tmp_path):
    with patch.object(web_module, "_vault_root", return_value=str(tmp_path)), \
         patch("pkms.web._watcher_ready_path", return_value=tmp_path / "no-watcher.ready"), \
         patch("pkms.web.handle_compile") as mock_compile:
        web_module._recover_uncompiled(CFG)  # no .search-index → no-op
    mock_compile.assert_not_called()


def test_wiki_graph_data_toggle_params(tmp_path):
    _make_graph_wiki(tmp_path)
    with _override(get_vault_root, lambda: str(tmp_path)):
        both_off = CLIENT.get("/wiki/graph/data?project=default&concepts=0&tags=0").json()
        assert {n["type"] for n in both_off["nodes"]} == {"article"}
        concepts_only = CLIENT.get("/wiki/graph/data?project=default&concepts=1&tags=0").json()
        types = {n["type"] for n in concepts_only["nodes"]}
        assert "concept" in types and "tag" not in types


def test_wiki_graph_concept_dedup(tmp_path):
    arts = tmp_path / "vault" / "default" / "wiki" / "articles"
    arts.mkdir(parents=True)
    (arts / "a.md").write_text("---\ntitle: A\n---\nSee [[energy-based-model]].\n")
    (arts / "b.md").write_text("---\ntitle: B\n---\nSee [[Energy Based Model]] and [[energy-based model]].\n")
    with _override(get_vault_root, lambda: str(tmp_path)):
        d = CLIENT.get("/wiki/graph/data?project=default").json()
    concepts = [n for n in d["nodes"] if n["type"] == "concept"]
    assert len(concepts) == 1                       # 3 variants collapse to 1 node
    assert concepts[0]["id"] == "concept:energy-based-model"


# ── B4 guards: web layer maps AccessDenied → 403 ───────────────────────────────

def test_ui_query_forbidden_for_non_member_on_claimed_project(tmp_path):
    from pkms.db import init_db, add_member
    (tmp_path / "vault").mkdir()
    db = str(tmp_path / "vault" / ".search-index")
    init_db(db)
    add_member(db, "demo", "alice", "owner", "t0")   # project is now claimed
    with _override(get_vault_root, lambda: str(tmp_path)), \
         _override(get_config, lambda: CFG):
        # bob (via the trusted header) is not a member → denied before any query runs
        resp = CLIENT.post("/ui/query",
                           data={"text": "q?", "user_id": "bob", "project": "demo"},
                           headers={"X-Auth-User": "bob"})
    assert resp.status_code == 403
    assert "Forbidden" in resp.text


# ── D1: conversation session cookie ────────────────────────────────────────────

def test_query_page_sets_session_cookie_once():
    from fastapi.testclient import TestClient as _TC
    c = _TC(app)
    r1 = c.get("/query")
    assert r1.status_code == 200
    sid = c.cookies.get("pkms_session")
    assert sid  # minted on first GET
    # second GET with the cookie present must NOT rotate it
    c.get("/query")
    assert c.cookies.get("pkms_session") == sid


def test_ui_query_threads_session_id_from_cookie(tmp_path):
    from fastapi.testclient import TestClient as _TC
    c = _TC(app)
    c.cookies.set("pkms_session", "sess-abc")
    with _override(get_vault_root, lambda: str(tmp_path)), \
         _override(get_config, lambda: CFG), \
         patch("pkms.web.handle_query", return_value=QUERY_RESULT) as mock_q:
        r = c.post("/ui/query", data={"text": "follow up?", "user_id": "alice"})
    assert r.status_code == 200
    assert mock_q.call_args.kwargs["session_id"] == "sess-abc"


def test_ui_query_session_id_none_when_cookie_absent(tmp_path):
    from fastapi.testclient import TestClient as _TC
    c = _TC(app)  # no GET, no cookie
    with _override(get_vault_root, lambda: str(tmp_path)), \
         _override(get_config, lambda: CFG), \
         patch("pkms.web.handle_query", return_value=QUERY_RESULT) as mock_q:
        r = c.post("/ui/query", data={"text": "q?", "user_id": "alice"})
    assert r.status_code == 200
    assert mock_q.call_args.kwargs["session_id"] is None
