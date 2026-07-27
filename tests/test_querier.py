import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from pkms.querier import (
    _fit_prompt,
    _format_hits,
    _greedy_fit,
    _index_slug_only,
    _mem0_recall,
    _mem0_store,
    _parse_llm_response,
    _reduce_index,
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


# ── 1.13 prompt budgeting ───────────────────────────────────────────────────────

def _big_index(n: int) -> str:
    return "\n".join(f"- [[art-{i}]] — Summary sentence number {i} about a topic." for i in range(n)) + "\n"


def test_greedy_fit_keeps_prefix_within_budget():
    items = ["aaaa", "bbbb", "cccc"]  # ~1 token each (len//4)
    kept, used = _greedy_fit(items, lambda s: len(s) // 4 or 1, budget=2)
    assert kept == ["aaaa", "bbbb"]
    assert used <= 2


def test_greedy_fit_empty_and_none():
    assert _greedy_fit([], len, 10) == ([], 0)
    assert _greedy_fit(None, len, 10) == ([], 0)


def test_index_slug_only_drops_summaries_keeps_all_slugs():
    idx = "- [[alpha]] — long summary here\n- [[beta]] — another summary\n"
    out = _index_slug_only(idx)
    assert out == "- [[alpha]]\n- [[beta]]"
    assert "summary" not in out


def test_reduce_index_untouched_when_it_fits():
    idx = "- [[alpha]] — s\n"
    fitted, action = _reduce_index(idx, budget=10_000)
    assert fitted == idx
    assert action is None


def test_reduce_index_falls_back_to_slug_only():
    idx = _big_index(50)
    # Budget big enough for slug-only but not the full summaries.
    slug_tokens = len(_index_slug_only(idx)) // 4
    fitted, action = _reduce_index(idx, budget=slug_tokens + 5)
    assert action == "slug_only"
    assert "Summary sentence" not in fitted
    assert "[[art-0]]" in fitted and "[[art-49]]" in fitted


def test_reduce_index_truncates_with_marker_when_tiny_budget():
    idx = _big_index(200)
    fitted, action = _reduce_index(idx, budget=20)
    assert action.startswith("truncated")
    assert "more articles omitted" in fitted
    assert "[[art-0]]" in fitted  # highest-priority (first) slugs survive


def test_reduce_index_dropped_when_no_room():
    fitted, action = _reduce_index(_big_index(10), budget=0)
    assert action == "dropped"
    assert "omitted" in fitted


def test_reduce_index_ignores_placeholder():
    fitted, action = _reduce_index("(wiki index not available)", budget=1)
    assert action is None


def test_fit_prompt_drops_index_before_chunks():
    # A pathologically large index must be trimmed while both hits survive.
    wiki, raw, prior, index, dropped = _fit_prompt(
        "question", "(none)", WIKI_HITS, RAW_HITS, [], _big_index(500),
        {"query": {"max_prompt_tokens": 1500}},
    )
    assert len(wiki) == len(WIKI_HITS)   # curated hits preserved
    assert len(raw) == len(RAW_HITS)
    assert "index" in dropped            # index was the thing cut
    assert "wiki_hits" not in dropped


def test_fit_prompt_noop_when_everything_fits():
    wiki, raw, prior, index, dropped = _fit_prompt(
        "q", "(none)", WIKI_HITS, RAW_HITS, ["one prior line"], "- [[alpha]] — s\n",
        {"query": {"max_prompt_tokens": 6000}},
    )
    assert dropped == {}
    assert index == "- [[alpha]] — s\n"
    assert prior == ["one prior line"]


def test_fit_prompt_drops_lowest_priority_first():
    # Budget sized to fit the fixed scaffold + exactly the wiki hit, nothing more:
    # wiki (highest priority) must survive while raw + prior + index are dropped.
    from pkms.querier import _SYNTHESIS_PROMPT, _estimate_tokens
    fixed = _estimate_tokens(_SYNTHESIS_PROMPT) + _estimate_tokens("q") + _estimate_tokens("(none)")
    wiki_cost = _estimate_tokens(_format_hits([WIKI_HITS[0]]))
    budget = fixed + wiki_cost + 2  # room for the wiki hit, not the raw hit
    wiki, raw, prior, _, dropped = _fit_prompt(
        "q", "(none)", WIKI_HITS, RAW_HITS, [], _big_index(100),
        {"query": {"max_prompt_tokens": budget}},
    )
    assert len(wiki) == len(WIKI_HITS)      # curated wiki (highest priority) preserved
    assert raw == []                        # raw dropped before touching wiki
    assert dropped["raw_hits"] == len(RAW_HITS)
    assert "index" in dropped               # index (lowest priority) reduced/dropped
    assert "wiki_hits" not in dropped


def test_query_trims_large_index_end_to_end(tmp_path):
    (tmp_path / "vault" / "default" / "wiki" / "articles").mkdir(parents=True)
    (tmp_path / "vault" / "default" / "outputs").mkdir(parents=True)
    (tmp_path / "vault" / "default" / "wiki" / "_index.md").write_text(_big_index(1000))
    captured = []

    def capture(agent, prompt, config, system=None):
        captured.append(prompt)
        return LLM_ANSWER

    cfg = {**CONFIG, "query": {**CONFIG["query"], "max_prompt_tokens": 1200}}
    with patch("pkms.querier.embed", return_value=[0.1, 0.2, 0.3, 0.4]), \
         patch("pkms.querier.search", side_effect=[WIKI_HITS, RAW_HITS]), \
         patch("pkms.querier.complete", side_effect=capture), \
         patch("pkms.querier._mem0_recall", return_value=[]), \
         patch("pkms.querier._mem0_store"), \
         patch("pkms.querier.upsert"):
        query("What is a Transformer?", "alice", str(tmp_path), cfg, session_id="big")

    # The full 1000-line index would blow the budget; the assembled prompt must
    # stay bounded and the wiki chunk must survive.
    assert len(captured[0]) // 4 < 1200 + 500  # prompt tokens under budget + slack
    assert "self-attention" in captured[0]     # wiki chunk preserved
