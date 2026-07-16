import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

import httpx

from pkms.ingestor import (
    chunk, extract_metadata, ingest, parse, _hash_file,
    fetch_and_ingest, _rewrite_arxiv_url, _derive_filename,
)


CONFIG = {
    "qdrant": {
        "host": "localhost",
        "port": 6333,
        "collections": {"raw": "raw", "wiki": "wiki", "outputs": "outputs"},
    },
    "embedding": {"dimension": 4},
    "chunking": {"max_tokens": 50},
    "ollama": {"host": "localhost", "port": 11434,
                "models": {"embedding": "nomic-embed-text"}},
    "llm_router": {
        "agents": {"ingestor": "ollama"},
        "fallback": {},
        "models": {"ollama": "mistral:7b", "claude": "claude-sonnet-4-6"},
    },
}


# ── parse ─────────────────────────────────────────────────────────────────────

def test_parse_txt(tmp_path):
    f = tmp_path / "doc.txt"
    f.write_text("hello world")
    assert parse(str(f)) == "hello world"


def test_parse_md(tmp_path):
    f = tmp_path / "doc.md"
    f.write_text("# Title\n\nBody text.")
    assert "Body text." in parse(str(f))


def test_parse_html(tmp_path):
    f = tmp_path / "doc.html"
    f.write_text("<html><body><h1>Title</h1><p>Hello</p></body></html>")
    result = parse(str(f))
    assert "Title" in result
    assert "Hello" in result
    assert "<" not in result


# ── chunk ─────────────────────────────────────────────────────────────────────

def test_chunk_splits_long_text():
    # Each paragraph ~10 tokens (40 chars); max_tokens=15 → forces splits
    text = "A" * 40 + "\n\n" + "B" * 40 + "\n\n" + "C" * 40
    chunks = chunk(text, max_tokens=15)
    assert len(chunks) == 3


def test_chunk_merges_short_paragraphs():
    text = "Short.\n\nAlso short."
    chunks = chunk(text, max_tokens=100)
    assert len(chunks) == 1


def test_chunk_detects_section_heading():
    text = "## Introduction\n\nSome content here."
    chunks = chunk(text, max_tokens=100)
    assert chunks[0]["section_heading"] == "Introduction"


def test_chunk_fallback_on_single_block():
    text = "No double newlines here at all."
    chunks = chunk(text, max_tokens=5)
    assert len(chunks) >= 1
    assert chunks[0]["text"]


# ── extract_metadata ──────────────────────────────────────────────────────────

def test_extract_metadata_parses_llm_json(tmp_path):
    meta_json = json.dumps({
        "title": "Test Doc",
        "summary_1line": "A test document.",
        "tags": ["test", "demo"],
        "doc_type": "note",
    })
    with patch("pkms.ingestor.complete", return_value=meta_json):
        result = extract_metadata("Some text", "vault/default/raw/doc.txt", CONFIG)
    assert result["title"] == "Test Doc"
    assert result["tags"] == ["test", "demo"]


def test_extract_metadata_handles_bad_json(tmp_path):
    with patch("pkms.ingestor.complete", return_value="not json at all"):
        result = extract_metadata("text", "vault/default/raw/doc.txt", CONFIG)
    assert "title" in result  # defaults applied
    assert "tags" in result


def test_extract_metadata_strips_code_fences():
    meta_json = '```json\n{"title":"T","summary_1line":"S","tags":[],"doc_type":"note"}\n```'
    with patch("pkms.ingestor.complete", return_value=meta_json):
        result = extract_metadata("text", "vault/default/raw/doc.txt", CONFIG)
    assert result["title"] == "T"


def test_extract_metadata_nonfatal_when_llm_raises():
    # Both backends failed (complete() raised) — must NOT kill the ingest;
    # fall back to filename-derived defaults instead.
    from pkms.llm import LLMError
    with patch("pkms.ingestor.complete", side_effect=LLMError("both backends down")):
        result = extract_metadata("text", "vault/default/raw/my_paper.pdf", CONFIG)
    assert result["title"] == "my_paper"      # filename stem
    assert result["tags"] == []
    assert result["doc_type"] == "pdf"


# ── ingest happy path ─────────────────────────────────────────────────────────

def _make_file(tmp_path, name="doc.txt", content="Hello.\n\nWorld."):
    f = tmp_path / "vault" / "default" / "raw" / name
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(content)
    return f


def test_ingest_returns_done(tmp_path):
    f = _make_file(tmp_path)
    db_path = str(tmp_path / "idx")
    from pkms.db import init_db
    init_db(db_path)

    meta_json = json.dumps({"title": "T", "summary_1line": "S",
                            "tags": ["a"], "doc_type": "note"})

    with patch("pkms.ingestor.complete", return_value=meta_json), \
         patch("pkms.embed._embed_batch", side_effect=lambda ts, c: [[0.1, 0.2, 0.3, 0.4]] * len(ts)), \
         patch("pkms.indexing.upsert_batch") as mock_upsert:
        result = ingest("vault/default/raw/doc.txt", str(tmp_path), db_path, CONFIG)

    assert result["status"] == "DONE"
    assert result["n_chunks"] >= 1
    assert mock_upsert.called


def test_ingest_stores_chunk_text_in_raw_payload(tmp_path):
    """Regression: the raw Qdrant payload MUST carry the chunk text — compiler
    _retrieve_chunks and querier _format_hits read payload['text']. Without it the
    wiki is compiled from headings only and query answers are ungrounded."""
    _make_file(tmp_path)
    db_path = str(tmp_path / "idx")
    from pkms.db import init_db
    init_db(db_path)

    meta_json = json.dumps({"title": "T", "summary_1line": "S", "tags": [], "doc_type": "note"})
    with patch("pkms.ingestor.complete", return_value=meta_json), \
         patch("pkms.embed._embed_batch", side_effect=lambda ts, c: [[0.1, 0.2, 0.3, 0.4]] * len(ts)), \
         patch("pkms.indexing.upsert_batch") as mock_upsert:
        ingest("vault/default/raw/doc.txt", str(tmp_path), db_path, CONFIG)

    payloads = [pl for c in mock_upsert.call_args_list for (_pid, _vec, pl) in c.args[1]]
    assert payloads
    assert all(p.get("text") for p in payloads)                 # every chunk has non-empty text
    joined = " ".join(p["text"] for p in payloads)
    assert "Hello" in joined and "World" in joined              # the actual document content


def test_ingest_skips_when_hash_unchanged(tmp_path):
    f = _make_file(tmp_path)
    db_path = str(tmp_path / "idx")
    from pkms.db import init_db, upsert_file
    init_db(db_path)

    from pkms.ingestor import _hash_file
    h = _hash_file(str(f))
    upsert_file(db_path, path="vault/default/raw/doc.txt", hash=h,
                qdrant_ids=[], collection="raw", indexed_at="2026-06-11T10:00:00Z",
                project="default")

    with patch("pkms.embed._embed_batch") as mock_embed:
        result = ingest("vault/default/raw/doc.txt", str(tmp_path), db_path, CONFIG)

    assert result["status"] == "SKIPPED"
    mock_embed.assert_not_called()


def test_ingest_result_includes_content_hash(tmp_path):
    f = _make_file(tmp_path)
    db_path = str(tmp_path / "idx")
    from pkms.db import init_db
    from pkms.ingestor import _hash_text
    init_db(db_path)

    meta_json = json.dumps({"title": "T", "summary_1line": "S", "tags": [], "doc_type": "note"})
    with patch("pkms.ingestor.complete", return_value=meta_json), \
         patch("pkms.embed._embed_batch", side_effect=lambda ts, c: [[0.1, 0.2, 0.3, 0.4]] * len(ts)), \
         patch("pkms.indexing.upsert_batch"):
        result = ingest("vault/default/raw/doc.txt", str(tmp_path), db_path, CONFIG)

    assert result["content_hash"] == _hash_text(parse(str(f)))


def test_ingest_skips_when_content_unchanged_despite_byte_drift(tmp_path):
    """Dynamic HTML re-fetch drifts the raw bytes but not the extracted text —
    the byte fast-path misses, the content hash matches → skip (no re-embed)."""
    f = _make_file(tmp_path)
    db_path = str(tmp_path / "idx")
    from pkms.db import init_db, upsert_file
    from pkms.ingestor import _hash_text
    init_db(db_path)

    # Seed with a STALE byte hash (so the fast path can't match) but the CURRENT
    # content hash (so the content path skips).
    upsert_file(db_path, path="vault/default/raw/doc.txt",
                hash="sha256:stale-bytes-from-a-previous-fetch",
                content_hash=_hash_text(parse(str(f))),
                qdrant_ids=["keep-me"], collection="raw",
                indexed_at="2026-06-11T10:00:00Z", project="default")

    with patch("pkms.ingestor.complete") as mock_complete, \
         patch("pkms.embed._embed_batch") as mock_embed, \
         patch("pkms.ingestor.delete_by_ids") as mock_delete:
        result = ingest("vault/default/raw/doc.txt", str(tmp_path), db_path, CONFIG)

    assert result["status"] == "SKIPPED"
    mock_embed.assert_not_called()      # no re-embed
    mock_delete.assert_not_called()     # old points left intact
    mock_complete.assert_not_called()   # skipped before metadata extraction


def test_ingest_force_reembeds_unchanged_file(tmp_path):
    # identical byte hash would normally skip; force=True must re-embed anyway
    f = _make_file(tmp_path)
    db_path = str(tmp_path / "idx")
    from pkms.db import init_db, upsert_file
    from pkms.ingestor import _hash_file
    init_db(db_path)
    h = _hash_file(str(f))
    upsert_file(db_path, path="vault/default/raw/doc.txt", hash=h,
                content_hash="sha256:whatever", qdrant_ids=["old-id"], collection="raw",
                indexed_at="2026-06-11T10:00:00Z", project="default")

    meta_json = json.dumps({"title": "T", "summary_1line": "S", "tags": [], "doc_type": "note"})
    with patch("pkms.ingestor.complete", return_value=meta_json), \
         patch("pkms.embed._embed_batch", side_effect=lambda ts, c: [[0.1, 0.2, 0.3, 0.4]] * len(ts)), \
         patch("pkms.indexing.upsert_batch") as mock_upsert, \
         patch("pkms.ingestor.delete_by_ids") as mock_delete:
        result = ingest("vault/default/raw/doc.txt", str(tmp_path), db_path, CONFIG, force=True)

    assert result["status"] == "DONE"
    assert mock_upsert.called
    mock_delete.assert_called_once_with("raw", ["old-id"], CONFIG)   # old points replaced


def test_ingest_reembeds_when_content_hash_differs(tmp_path):
    # bytes changed AND parsed text changed → content path must NOT skip
    f = _make_file(tmp_path)
    db_path = str(tmp_path / "idx")
    from pkms.db import init_db, upsert_file
    init_db(db_path)
    upsert_file(db_path, path="vault/default/raw/doc.txt",
                hash="sha256:old", content_hash="sha256:some-other-content",
                qdrant_ids=["old-id"], collection="raw",
                indexed_at="2026-06-11T10:00:00Z", project="default")

    meta_json = json.dumps({"title": "T", "summary_1line": "S", "tags": [], "doc_type": "note"})
    with patch("pkms.ingestor.complete", return_value=meta_json), \
         patch("pkms.embed._embed_batch", side_effect=lambda ts, c: [[0.1, 0.2, 0.3, 0.4]] * len(ts)), \
         patch("pkms.indexing.upsert_batch"), \
         patch("pkms.ingestor.delete_by_ids") as mock_delete:
        result = ingest("vault/default/raw/doc.txt", str(tmp_path), db_path, CONFIG)

    assert result["status"] == "DONE"
    mock_delete.assert_called_once_with("raw", ["old-id"], CONFIG)


def test_ingest_deletes_old_points_on_rehash(tmp_path):
    f = _make_file(tmp_path)
    db_path = str(tmp_path / "idx")
    from pkms.db import init_db, upsert_file
    init_db(db_path)
    upsert_file(db_path, path="vault/default/raw/doc.txt", hash="sha256:old",
                qdrant_ids=["old-id"], collection="raw", indexed_at="2026-06-11T10:00:00Z",
                project="default")

    meta_json = json.dumps({"title": "T", "summary_1line": "S",
                            "tags": [], "doc_type": "note"})
    with patch("pkms.ingestor.complete", return_value=meta_json), \
         patch("pkms.embed._embed_batch", side_effect=lambda ts, c: [[0.1, 0.2, 0.3, 0.4]] * len(ts)), \
         patch("pkms.indexing.upsert_batch"), \
         patch("pkms.ingestor.delete_by_ids") as mock_delete:
        ingest("vault/default/raw/doc.txt", str(tmp_path), db_path, CONFIG)

    mock_delete.assert_called_once_with("raw", ["old-id"], CONFIG)


def test_ingest_raises_on_missing_file(tmp_path):
    db_path = str(tmp_path / "idx")
    from pkms.db import init_db
    init_db(db_path)
    with pytest.raises(FileNotFoundError):
        ingest("vault/default/raw/missing.txt", str(tmp_path), db_path, CONFIG)


# ── _rewrite_arxiv_url ────────────────────────────────────────────────────────

def test_rewrite_arxiv_abs_to_pdf():
    url = "https://arxiv.org/abs/1706.03762"
    assert _rewrite_arxiv_url(url) == "https://arxiv.org/pdf/1706.03762"


def test_rewrite_non_arxiv_unchanged():
    url = "https://example.com/paper.pdf"
    assert _rewrite_arxiv_url(url) == url


# ── _derive_filename ──────────────────────────────────────────────────────────

def _mock_resp(content_disposition="", content_type="text/html"):
    r = MagicMock()
    r.headers = {"content-disposition": content_disposition,
                 "content-type": content_type}
    return r


def test_derive_filename_from_content_disposition():
    resp = _mock_resp(content_disposition='attachment; filename="attention.pdf"')
    assert _derive_filename("https://example.com/dl", resp) == "attention.pdf"


def test_derive_filename_from_url_path():
    resp = _mock_resp()
    assert _derive_filename("https://arxiv.org/pdf/1706.03762", resp) == "1706.03762.html"


def test_derive_filename_pdf_content_type():
    resp = _mock_resp(content_type="application/pdf")
    name = _derive_filename("https://example.com/download", resp)
    assert name.endswith(".pdf")


def test_derive_filename_sanitises_special_chars():
    resp = _mock_resp(content_disposition='filename="weird file name!.pdf"')
    name = _derive_filename("https://x.com", resp)
    assert " " not in name and "!" not in name


# ── fetch_and_ingest ──────────────────────────────────────────────────────────

CONFIG_FETCH = {**CONFIG, "fetch": {"timeout_seconds": 10, "user_agent": "pkms/0.1", "arxiv_rewrite": True}}


def test_fetch_and_ingest_saves_file_and_ingests(tmp_path):
    (tmp_path / "vault" / "default" / "raw").mkdir(parents=True)
    db_path = str(tmp_path / "idx")
    from pkms.db import init_db
    init_db(db_path)

    fake_resp = MagicMock()
    fake_resp.headers = {"content-disposition": 'filename="fetched.txt"',
                         "content-type": "text/plain"}
    fake_resp.content = b"Hello.\n\nFetched world."
    fake_resp.raise_for_status = MagicMock()

    meta_json = json.dumps({"title": "Fetched", "summary_1line": "S",
                            "tags": ["fetch"], "doc_type": "note"})

    with patch("pkms.ingestor.httpx.get", return_value=fake_resp), \
         patch("pkms.ingestor.complete", return_value=meta_json), \
         patch("pkms.embed._embed_batch", side_effect=lambda ts, c: [[0.1, 0.2, 0.3, 0.4]] * len(ts)), \
         patch("pkms.indexing.upsert_batch"):
        result = fetch_and_ingest(
            "https://example.com/paper", str(tmp_path), db_path, CONFIG_FETCH
        )

    assert result["status"] == "DONE"
    assert (tmp_path / "vault" / "default" / "raw" / "fetched.txt").exists()


def test_fetch_and_ingest_rewrites_arxiv_url(tmp_path):
    (tmp_path / "vault" / "default" / "raw").mkdir(parents=True)
    db_path = str(tmp_path / "idx")
    from pkms.db import init_db
    init_db(db_path)

    fake_resp = MagicMock()
    fake_resp.headers = {"content-disposition": "", "content-type": "application/pdf"}
    fake_resp.content = b"%PDF fake"
    fake_resp.raise_for_status = MagicMock()

    meta_json = json.dumps({"title": "T", "summary_1line": "S", "tags": [], "doc_type": "pdf"})

    captured_urls = []

    def mock_get(url, **kwargs):
        captured_urls.append(url)
        return fake_resp

    with patch("pkms.ingestor.httpx.get", side_effect=mock_get), \
         patch("pkms.ingestor.parse", return_value="Fake text.\n\nSecond paragraph."), \
         patch("pkms.ingestor.complete", return_value=meta_json), \
         patch("pkms.embed._embed_batch", side_effect=lambda ts, c: [[0.1, 0.2, 0.3, 0.4]] * len(ts)), \
         patch("pkms.indexing.upsert_batch"):
        fetch_and_ingest(
            "https://arxiv.org/abs/1706.03762", str(tmp_path), db_path, CONFIG_FETCH
        )

    assert "arxiv.org/pdf/" in captured_urls[0]


def test_fetch_and_ingest_raises_on_http_error(tmp_path):
    db_path = str(tmp_path / "idx")
    from pkms.db import init_db
    init_db(db_path)

    req = httpx.Request("GET", "https://example.com/missing")
    err404 = httpx.HTTPStatusError("404", request=req,
                                   response=httpx.Response(404, request=req))
    with patch("pkms.ingestor.httpx.get", side_effect=err404) as mg:
        with pytest.raises(httpx.HTTPStatusError):
            fetch_and_ingest("https://example.com/missing", str(tmp_path), db_path, CONFIG_FETCH)
    assert mg.call_count == 1   # 404 is permanent — no retry burned on it


def test_extract_metadata_uses_low_temperature():
    import json as _json
    meta = _json.dumps({"title": "T", "summary_1line": "S", "tags": [], "doc_type": "note"})
    with patch("pkms.ingestor.complete", return_value=meta) as m:
        extract_metadata("text", "vault/default/raw/doc.txt", CONFIG)
    assert m.call_args.kwargs.get("temperature") == 0.1


def test_extract_metadata_uses_small_model_and_cap():
    import json as _json
    cfg = {**CONFIG, "ingest": {"metadata_model": "llama3.2:1b",
                                 "metadata_num_predict": 200, "metadata_temperature": 0.1}}
    meta = _json.dumps({"title": "T", "summary_1line": "S", "tags": [], "doc_type": "note"})
    with patch("pkms.ingestor.complete", return_value=meta) as m:
        extract_metadata("text", "vault/default/raw/doc.txt", cfg)
    kw = m.call_args.kwargs
    assert kw["model"] == "llama3.2:1b"
    assert kw["num_predict"] == 200
    assert kw["retries"] == 1


def test_chunk_hard_splits_oversized_single_paragraph():
    # A blob with NO blank lines, far over max_tokens, must split into multiple
    # in-budget chunks (regression: pdfminer one-blob output → 1 giant chunk → embed 500).
    text = "word " * 4000   # ~20k chars, ~5k tokens, zero '\n\n'
    chunks = chunk(text, max_tokens=512)
    assert len(chunks) > 1
    assert all(len(c["text"]) <= 512 * 4 for c in chunks)  # none exceeds the token budget


def test_fetch_and_ingest_retries_on_429_with_retry_after(tmp_path):
    (tmp_path / "vault" / "default" / "raw").mkdir(parents=True)
    db_path = str(tmp_path / "idx")
    from pkms.db import init_db
    init_db(db_path)

    fake_resp = MagicMock()
    fake_resp.headers = {"content-disposition": 'filename="fetched.txt"', "content-type": "text/plain"}
    fake_resp.content = b"Hello.\n\nFetched world."
    fake_resp.raise_for_status = MagicMock()

    req = httpx.Request("GET", "https://example.com/paper")
    resp429 = httpx.Response(429, request=req, headers={"retry-after": "7"})
    err = httpx.HTTPStatusError("429", request=req, response=resp429)
    meta_json = json.dumps({"title": "F", "summary_1line": "S", "tags": [], "doc_type": "note"})

    with patch("pkms.ingestor.httpx.get", side_effect=[err, fake_resp]) as mg, \
         patch("pkms.llm.time.sleep") as ms, \
         patch("pkms.ingestor.complete", return_value=meta_json), \
         patch("pkms.embed._embed_batch", side_effect=lambda ts, c: [[0.1, 0.2, 0.3, 0.4]] * len(ts)), \
         patch("pkms.indexing.upsert_batch"):
        result = fetch_and_ingest("https://example.com/paper", str(tmp_path), db_path, CONFIG_FETCH)

    assert result["status"] == "DONE"
    assert mg.call_count == 2           # retried after the 429
    ms.assert_called_once_with(7.0)     # honoured Retry-After


# ── B6: extraction quality (glued-PDF fallback + HTML main-content) ───────────

import pkms.ingestor as _ing


def test_space_ratio_and_looks_glued():
    assert _ing._looks_glued("Neuro" + "SymbolicAIin2024xyz" * 30) is True
    assert _ing._looks_glued("This is normal prose with many spaces. " * 30) is False
    assert _ing._looks_glued("short") is False   # length floor


def test_parse_pdf_falls_back_to_pypdf_when_glued():
    glued = "Neuro-SymbolicAIin2024andbeyondwithnowordspacesatall" * 30
    spaced = "Neuro Symbolic AI in 2024 and beyond with proper spacing here. " * 30
    with patch("pdfminer.high_level.extract_text", return_value=glued), \
         patch("pkms.ingestor._parse_pdf_pypdf", return_value=spaced):
        out = _ing._parse_pdf("x.pdf")
    assert out == spaced   # better-spaced extraction wins


def test_parse_pdf_keeps_pdfminer_when_clean():
    good = "This is a normal sentence with plenty of spaces in it. " * 20
    with patch("pdfminer.high_level.extract_text", return_value=good), \
         patch("pkms.ingestor._parse_pdf_pypdf") as mock_pypdf:
        out = _ing._parse_pdf("x.pdf")
    assert out == good
    mock_pypdf.assert_not_called()   # no fallback when spacing is fine


_HTML_FIXTURE = """<html><head><title>T</title>
<script>var x = {a:1}; doEvilTracking();</script>
<style>.nav{color:red}</style></head>
<body>
<nav>Home About Login Signup</nav>
<article><h1>Contrastive Learning</h1>
<p>InfoNCE is a contrastive loss used in self-supervised representation learning.
It maximises agreement between positive pairs while pushing negative samples apart,
and a temperature parameter controls the sharpness of the resulting distribution.</p>
</article>
<footer>Copyright 2026 Privacy Terms</footer>
</body></html>"""


def test_parse_html_extracts_article_drops_boilerplate(tmp_path):
    f = tmp_path / "page.html"; f.write_text(_HTML_FIXTURE, encoding="utf-8")
    out = _ing._parse_html(str(f))
    assert "InfoNCE is a contrastive loss" in out   # article body kept
    assert "doEvilTracking" not in out              # inline JS dropped
    assert "color:red" not in out                   # inline CSS dropped


def test_parse_html_bs4_fallback_drops_script(tmp_path):
    f = tmp_path / "page.html"; f.write_text(_HTML_FIXTURE, encoding="utf-8")
    with patch("trafilatura.extract", return_value=None):   # force the bs4 fallback path
        out = _ing._parse_html(str(f))
    assert "doEvilTracking" not in out
    assert "InfoNCE is a contrastive loss" in out
