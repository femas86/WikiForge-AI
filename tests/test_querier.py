import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from pkms.querier import (
    _format_hits,
    _mem0_recall,
    _mem0_store,
    _parse_llm_response,
    query,
)


CONFIG = {
    "qdrant": {
        "host": "localhost",
        "port": 6333,
        "collections": {"raw": "raw", "wiki": "wiki", "outputs": "outputs"},
    },
    "embedding": {"dimension": 4},
    "chunking": {"max_tokens": 200},
    "ollama": {"host": "localhost", "port": 11434,
                "models": {"embedding": "nomic-embed-text"}},
    "llm_router": {
        "agents": {"querier": "claude"},
        "fallback": {"claude": "ollama"},
        "models": {"claude": "claude-sonnet-4-6", "ollama": "mistral:7b"},
    },
    "query": {"top_k_wiki": 3, "top_k_raw": 3, "mem0_recall": 2},
}

WIKI_HITS = [
    {"id": "w1", "score": 0.95, "payload": {
        "path": "vault/default/wiki/articles/transformers.md",
        "chunk_index": 0, "section_heading": "Introduction",
        "text": "Transformers use self-attention.",
    }},
]
RAW_HITS = [
    {"id": "r1", "score": 0.80, "payload": {
        "path": "vault/default/raw/attention.pdf",
        "chunk_index": 2, "section_heading": "Model Architecture",
        "text": "The encoder has 6 layers.",
    }},
]

LLM_ANSWER = """\
The Transformer architecture uses self-attention mechanisms [[transformers]].
The encoder consists of 6 stacked layers (raw:vault/default/raw/attention.pdf#2).

```json
{"sources": ["vault/default/wiki/articles/transformers.md", "vault/default/raw/attention.pdf"], "coverage": "full"}
```"""


# ── _format_hits ──────────────────────────────────────────────────────────────

def test_format_hits_includes_path_and_text():
    result = _format_hits(WIKI_HITS)
    assert "vault/default/wiki/articles/transformers.md" in result
    assert "self-attention" in result


def test_format_hits_empty():
    assert _format_hits([]) == "(none)"


def test_format_hits_includes_section_heading():
    result = _format_hits(WIKI_HITS)
    assert "Introduction" in result


# ── _parse_llm_response ───────────────────────────────────────────────────────

def test_parse_extracts_sources_and_coverage():
    answer, sources, coverage = _parse_llm_response(LLM_ANSWER)
    assert "self-attention" in answer
    assert "vault/default/wiki/articles/transformers.md" in sources
    assert coverage == "full"


def test_parse_strips_json_block_from_answer():
    answer, _, _ = _parse_llm_response(LLM_ANSWER)
    assert "```json" not in answer


def test_parse_fallback_on_bad_json():
    raw = "Here is the answer. No JSON block at all."
    answer, sources, coverage = _parse_llm_response(raw)
    assert answer == raw.strip()
    assert sources == []
    assert coverage == "partial"


def test_parse_coverage_none():
    raw = 'No relevant documents.\n```json\n{"sources": [], "coverage": "none"}\n```'
    _, _, coverage = _parse_llm_response(raw)
    assert coverage == "none"


# ── session id uniqueness & output payload ───────────────────────────────────

def _run_query(tmp_path, mock_upsert=None, session_id=None):
    with patch("pkms.querier.embed", return_value=[0.1, 0.2, 0.3, 0.4]), \
         patch("pkms.querier.search", side_effect=[WIKI_HITS, RAW_HITS]), \
         patch("pkms.querier.complete", return_value=LLM_ANSWER), \
         patch("pkms.querier._mem0_recall", return_value=[]), \
         patch("pkms.querier._mem0_store"), \
         patch("pkms.querier.upsert", mock_upsert or MagicMock()):
        return query("Q?", "default", str(tmp_path), CONFIG, session_id=session_id)


def test_query_session_ids_unique_within_same_second(tmp_path):
    (tmp_path / "vault" / "default" / "outputs").mkdir(parents=True)
    ids = set()
    for _ in range(3):
        with patch("pkms.querier.embed", return_value=[0.1, 0.2, 0.3, 0.4]), \
             patch("pkms.querier.search", side_effect=[WIKI_HITS, RAW_HITS]), \
             patch("pkms.querier.complete", return_value=LLM_ANSWER), \
             patch("pkms.querier._mem0_recall", return_value=[]), \
             patch("pkms.querier._mem0_store"), \
             patch("pkms.querier.upsert"):
            ids.add(query("Q?", "default", str(tmp_path), CONFIG)["session_id"])
    assert len(ids) == 3


def test_output_payload_has_hash_and_section_heading(tmp_path):
    (tmp_path / "vault" / "default" / "outputs").mkdir(parents=True)
    mock_upsert = MagicMock()
    _run_query(tmp_path, mock_upsert=mock_upsert, session_id="payload_test")

    assert mock_upsert.call_count >= 1
    payload = mock_upsert.call_args_list[0][0][3]
    assert payload["hash"].startswith("sha256:")
    assert len(payload["hash"]) == len("sha256:") + 64
    assert "section_heading" in payload
    assert payload["chunk_total"] >= 1


# ── query integration ─────────────────────────────────────────────────────────

def _setup_vault(tmp_path):
    (tmp_path / "vault" / "default" / "wiki" / "articles").mkdir(parents=True)
    (tmp_path / "vault" / "default" / "outputs").mkdir(parents=True)
    (tmp_path / "vault" / "default" / "wiki" / "_index.md").write_text(
        "- [[transformers]] — Article about transformers\n"
    )
    return tmp_path


def test_query_returns_answer(tmp_path):
    _setup_vault(tmp_path)

    with patch("pkms.querier.embed", return_value=[0.1, 0.2, 0.3, 0.4]), \
         patch("pkms.querier.search", side_effect=[WIKI_HITS, RAW_HITS]), \
         patch("pkms.querier.complete", return_value=LLM_ANSWER), \
         patch("pkms.querier._mem0_recall", return_value=[]), \
         patch("pkms.querier._mem0_store"), \
         patch("pkms.querier.upsert"):
        result = query("What is a Transformer?", "alice", str(tmp_path), CONFIG,
                       session_id="test_session")

    assert result["coverage"] == "full"
    assert "self-attention" in result["answer_md"]
    assert result["sources"] == ["vault/default/wiki/articles/transformers.md", "vault/default/raw/attention.pdf"]


def test_query_result_carries_phase_metrics(tmp_path):
    """The result must expose the D-axis provenance the F2 harness reads:
    a summary (llm_calls/tokens/durations) plus per-phase timed events."""
    _setup_vault(tmp_path)

    with patch("pkms.querier.embed", return_value=[0.1, 0.2, 0.3, 0.4]), \
         patch("pkms.querier.search", side_effect=[WIKI_HITS, RAW_HITS]), \
         patch("pkms.querier.complete", return_value=LLM_ANSWER), \
         patch("pkms.querier._mem0_recall", return_value=[]), \
         patch("pkms.querier._mem0_store"), \
         patch("pkms.querier.upsert"):
        result = query("What is a Transformer?", "alice", str(tmp_path), CONFIG,
                       session_id="test_session")

    m = result["metrics"]
    kinds = {e["kind"] for e in m["events"]}
    assert {"retrieval", "memory_recall", "synthesis", "memory_store"} <= kinds
    assert set(m["summary"]) == {"llm_calls", "tokens_in", "tokens_out", "durations_ms"}
    assert "synthesis" in m["summary"]["durations_ms"]


def test_query_writes_output_file(tmp_path):
    _setup_vault(tmp_path)

    with patch("pkms.querier.embed", return_value=[0.1, 0.2, 0.3, 0.4]), \
         patch("pkms.querier.search", side_effect=[WIKI_HITS, RAW_HITS]), \
         patch("pkms.querier.complete", return_value=LLM_ANSWER), \
         patch("pkms.querier._mem0_recall", return_value=[]), \
         patch("pkms.querier._mem0_store"), \
         patch("pkms.querier.upsert"):
        result = query("What is a Transformer?", "alice", str(tmp_path), CONFIG,
                       session_id="test_session")

    out_file = tmp_path / "vault" / "default" / "outputs" / "test_session.md"
    assert out_file.exists()
    content = out_file.read_text()
    assert "What is a Transformer?" in content
    assert "self-attention" in content


def test_query_reads_index_md(tmp_path):
    _setup_vault(tmp_path)
    captured_prompts = []

    def capture_complete(agent, prompt, config, system=None):
        captured_prompts.append(prompt)
        return LLM_ANSWER

    with patch("pkms.querier.embed", return_value=[0.1, 0.2, 0.3, 0.4]), \
         patch("pkms.querier.search", side_effect=[WIKI_HITS, RAW_HITS]), \
         patch("pkms.querier.complete", side_effect=capture_complete), \
         patch("pkms.querier._mem0_recall", return_value=[]), \
         patch("pkms.querier._mem0_store"), \
         patch("pkms.querier.upsert"):
        query("What is a Transformer?", "alice", str(tmp_path), CONFIG,
              session_id="s1")

    assert "transformers" in captured_prompts[0].lower()


def test_query_handles_missing_index_md(tmp_path):
    (tmp_path / "vault" / "default" / "wiki").mkdir(parents=True)
    (tmp_path / "vault" / "default" / "outputs").mkdir(parents=True)
    # No _index.md created

    with patch("pkms.querier.embed", return_value=[0.1, 0.2, 0.3, 0.4]), \
         patch("pkms.querier.search", side_effect=[[], []]), \
         patch("pkms.querier.complete", return_value=LLM_ANSWER), \
         patch("pkms.querier._mem0_recall", return_value=[]), \
         patch("pkms.querier._mem0_store"), \
         patch("pkms.querier.upsert"):
        result = query("anything", "bob", str(tmp_path), CONFIG, session_id="s2")

    assert "answer_md" in result


def test_query_uses_prior_context(tmp_path):
    _setup_vault(tmp_path)
    captured_prompts = []

    def capture(agent, prompt, config, system=None):
        captured_prompts.append(prompt)
        return LLM_ANSWER

    with patch("pkms.querier.embed", return_value=[0.1, 0.2, 0.3, 0.4]), \
         patch("pkms.querier.search", side_effect=[WIKI_HITS, RAW_HITS]), \
         patch("pkms.querier.complete", side_effect=capture), \
         patch("pkms.querier._mem0_store"), \
         patch("pkms.querier.upsert"):
        query("What is a Transformer?", "alice", str(tmp_path), CONFIG,
              session_id="s3", prior_context=["Alice asked about attention last week."])

    assert "Alice asked about attention last week." in captured_prompts[0]


def test_query_injects_user_style(tmp_path):
    _setup_vault(tmp_path)
    captured_prompts = []

    def capture(agent, prompt, config, system=None):
        captured_prompts.append(prompt)
        return LLM_ANSWER

    with patch("pkms.querier.embed", return_value=[0.1, 0.2, 0.3, 0.4]), \
         patch("pkms.querier.search", side_effect=[WIKI_HITS, RAW_HITS]), \
         patch("pkms.querier.complete", side_effect=capture), \
         patch("pkms.querier.load_user_style", return_value="ZZSTYLE: answer in Italian, TL;DR first"), \
         patch("pkms.querier._mem0_store"), \
         patch("pkms.querier.upsert"):
        query("What is a Transformer?", "alice", str(tmp_path), CONFIG,
              session_id="s6", prior_context=[])

    assert "ZZSTYLE: answer in Italian, TL;DR first" in captured_prompts[0]


def test_query_mem0_recall_failure_is_nonfatal(tmp_path):
    _setup_vault(tmp_path)

    with patch("pkms.querier.embed", return_value=[0.1, 0.2, 0.3, 0.4]), \
         patch("pkms.querier.search", side_effect=[WIKI_HITS, RAW_HITS]), \
         patch("pkms.querier.complete", return_value=LLM_ANSWER), \
         patch("pkms.querier._mem0_recall", side_effect=Exception("mem0 down")), \
         patch("pkms.querier._mem0_store"), \
         patch("pkms.querier.upsert"):
        # _mem0_recall is called only when prior_context is None and it's handled
        # inside the function — passing prior_context=[] bypasses it
        result = query("question", "alice", str(tmp_path), CONFIG,
                       session_id="s4", prior_context=[])

    assert result["answer_md"]


def test_query_mem0_store_failure_is_nonfatal(tmp_path):
    _setup_vault(tmp_path)

    with patch("pkms.querier.embed", return_value=[0.1, 0.2, 0.3, 0.4]), \
         patch("pkms.querier.search", side_effect=[WIKI_HITS, RAW_HITS]), \
         patch("pkms.querier.complete", return_value=LLM_ANSWER), \
         patch("pkms.querier._mem0_recall", return_value=[]), \
         patch("pkms.querier._mem0_store", side_effect=Exception("mem0 down")), \
         patch("pkms.querier.upsert"):
        result = query("question", "alice", str(tmp_path), CONFIG,
                       session_id="s5")

    assert result["coverage"] == "full"


# ── managed Mem0 client (app.mem0.ai) ─────────────────────────────────────────

def test_mem0_recall_parses_results_and_uses_filters():
    client = MagicMock()
    # Platform v1.1 response shape: {"results": [...]}; empty/blank memories dropped
    client.search.return_value = {
        "results": [{"memory": "m1"}, {"memory": "m2"}, {"memory": ""}, {}]
    }
    with patch("pkms.querier._mem0_client", return_value=client):
        out = _mem0_recall("what about attention?", "alice", CONFIG)

    assert out == ["m1", "m2"]
    # entity params must be in `filters`, and limit is `top_k`
    _, kwargs = client.search.call_args
    assert kwargs["filters"] == {"user_id": "alice"}
    assert kwargs["top_k"] == CONFIG["query"]["mem0_recall"]


def test_mem0_recall_handles_bare_list_response():
    client = MagicMock()
    client.search.return_value = [{"memory": "x"}]
    with patch("pkms.querier._mem0_client", return_value=client):
        assert _mem0_recall("q", "bob", CONFIG) == ["x"]


def test_mem0_recall_nonfatal_when_client_unavailable():
    # e.g. MEM0_API_KEY unset → MemoryClient() raises; recall must degrade to []
    with patch("pkms.querier._mem0_client", side_effect=Exception("no api key")):
        assert _mem0_recall("q", "alice", CONFIG) == []


def test_mem0_store_calls_add_with_user_and_metadata():
    client = MagicMock()
    with patch("pkms.querier._mem0_client", return_value=client):
        _mem0_store("Q?", "A.", ["src#0"], "alice", "sess1", CONFIG, project="og-mdai")

    _, kwargs = client.add.call_args
    assert kwargs["user_id"] == "alice"
    assert kwargs["metadata"]["type"] == "qa_interaction"
    assert kwargs["metadata"]["project"] == "og-mdai"
    assert kwargs["metadata"]["sources"] == ["src#0"]
