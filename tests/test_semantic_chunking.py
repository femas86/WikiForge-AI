from unittest.mock import patch

import pkms.ingestor as ing
from pkms.ingestor import (
    _breakpoint_threshold,
    _char_split,
    _cosine,
    _semantic_split,
    _split_sentences,
    chunk,
)

# semantic_min_tokens well below max_tokens so a "wide" paragraph triggers
# embedding analysis even when it fits under max_tokens (the whole point of F1).
SEM = {"chunking": {"strategy": "semantic", "max_tokens": 100,
                    "semantic_min_tokens": 5, "semantic_breakpoint_zscore": 1.0}}
FIXED = {"chunking": {"strategy": "fixed", "max_tokens": 15}}


def _fake_embed_two_topics():
    """'ALPHA' sentences embed one way, 'BETA' the other → a hard boundary between
    the two groups, ~zero cosine distance within each group."""
    def _embed(text, config):
        return [1.0, 0.0] if "ALPHA" in text else [0.0, 1.0]
    return _embed


# ── helpers ───────────────────────────────────────────────────────────────────

def _batch(per_text):
    """Adapt a per-text fake to the batch seam _embed_batch(texts, config)."""
    return lambda texts, config: [per_text(t, config) for t in texts]


def test_split_sentences():
    assert _split_sentences("One here. Two! Three?\nFourth line") == \
        ["One here.", "Two!", "Three?", "Fourth line"]


def test_cosine():
    assert _cosine([1.0, 0.0], [0.0, 1.0]) == 0.0
    assert abs(_cosine([1.0, 1.0], [2.0, 2.0]) - 1.0) < 1e-9
    assert _cosine([0.0, 0.0], [1.0, 1.0]) == 0.0   # zero-vector guard


def test_breakpoint_threshold_flags_outlier():
    # bimodal distances: the lone 1.0 must sit above mean+std; within-topic 0s below
    dists = [0.0, 0.0, 1.0, 0.0, 0.0]
    thr = _breakpoint_threshold(dists, 1.0)
    assert max(dists) > thr > min(dists)


def test_breakpoint_threshold_uniform_has_no_break():
    # all equal → std 0 → threshold == mean → strict '>' yields no breaks
    assert _breakpoint_threshold([0.3, 0.3, 0.3], 1.0) == 0.3
    assert _breakpoint_threshold([], 1.0) == float("inf")


# ── dispatch: 'fixed' (and no config) never embeds ────────────────────────────

def test_chunk_defaults_to_fixed_without_config():
    with patch("pkms.embed._embed_batch", side_effect=AssertionError("embed must not run")):
        out = chunk("a\n\nb\n\nc", 15)
    assert all("section_heading" in c and "text" in c for c in out)


def test_chunk_fixed_strategy_does_not_embed():
    with patch("pkms.embed._embed_batch", side_effect=AssertionError("embed must not run")):
        chunk("word " * 200, 15, FIXED)   # oversized but fixed → char split, no embed


# ── _semantic_split ───────────────────────────────────────────────────────────

def test_semantic_split_breaks_at_topic_shift():
    para = "ALPHA one. ALPHA two. ALPHA three. BETA four. BETA five. BETA six."
    with patch("pkms.embed._embed_batch", side_effect=_batch(_fake_embed_two_topics())):
        groups = _semantic_split(para, max_tokens=100, config=SEM)  # size irrelevant → topic drives split
    assert len(groups) == 2
    assert "ALPHA" in groups[0] and "BETA" not in groups[0]
    assert "BETA" in groups[1] and "ALPHA" not in groups[1]


def test_semantic_split_keeps_single_topic_together():
    para = "ALPHA one. ALPHA two. ALPHA three. ALPHA four."
    with patch("pkms.embed._embed_batch", side_effect=_batch(_fake_embed_two_topics())):
        groups = _semantic_split(para, max_tokens=100, config=SEM)
    assert groups == [para]   # no topic shift, fits budget → one chunk


def test_semantic_split_falls_back_to_char_on_embed_error():
    para = "First sentence. Second sentence. Third sentence."
    with patch("pkms.embed._embed_batch", side_effect=RuntimeError("ollama down")):
        assert _semantic_split(para, max_tokens=5, config=SEM) == _char_split(para, 5)


def test_semantic_split_single_short_sentence_skips_embedding():
    # one short unit → nothing to compare → char-split, no embedding call
    with patch("pkms.embed._embed_batch", side_effect=AssertionError("must not embed")):
        assert _semantic_split("Just one sentence.", max_tokens=100, config=SEM) == \
            _char_split("Just one sentence.", 100)


def test_semantic_split_caps_oversized_units_before_embedding():
    """Regression: a run-on 'sentence' (scraped HTML, no . ! ? or newline) must be
    char-split into embed-safe units BEFORE embedding — else Ollama /api/embeddings
    500s on the oversized input (the infonce.html failure)."""
    max_tokens = 10
    monster = "word" * 200                      # ~800 chars ≈ 200 tokens, one 'sentence'
    para = f"{monster}. short tail sentence here."
    seen: list[str] = []

    def _rec_embed(text, config):
        seen.append(text)
        return [1.0, 0.0]

    with patch("pkms.embed._embed_batch", side_effect=_batch(_rec_embed)):
        groups = _semantic_split(para, max_tokens, SEM)

    assert seen, "should have embedded the capped units"
    # every embedded unit stays within the embedder-safe budget
    assert all(ing._estimate_tokens(u) <= max_tokens for u in seen)
    assert all(ing._estimate_tokens(g) <= max_tokens for g in groups)


def test_hybrid_survives_embed_500_via_fallback():
    # embed raises (Ollama 500) → the wide block degrades to a char-split, ingest proceeds
    blob = "runon " * 300  # one oversized paragraph, no sentence breaks
    with patch("pkms.embed._embed_batch", side_effect=RuntimeError("500 Internal Server Error")):
        chunks = chunk(blob, 15, SEM)
    assert len(chunks) > 1
    assert all(ing._estimate_tokens(c["text"]) <= 15 for c in chunks)


def test_semantic_split_respects_token_budget():
    para = " ".join(f"Word{i} sentence text here." for i in range(20))
    with patch("pkms.embed._embed_batch", side_effect=lambda ts, c: [[1.0, 0.0]] * len(ts)):  # no topic shift → size drives it
        groups = _semantic_split(para, max_tokens=15, config=SEM)
    assert len(groups) > 1
    assert all(ing._estimate_tokens(g) <= 15 for g in groups)


# ── hybrid chunk(): the F1 behaviour the design targets ───────────────────────

def test_hybrid_splits_wide_paragraph_under_max_tokens():
    """A paragraph well under max_tokens but spanning two concepts must be split
    by embeddings — the case a size-only failsafe would have kept whole."""
    para = "ALPHA one. ALPHA two. ALPHA three. BETA four. BETA five. BETA six."
    assert ing._estimate_tokens(para) < SEM["chunking"]["max_tokens"]   # NOT oversized
    with patch("pkms.embed._embed_batch", side_effect=_batch(_fake_embed_two_topics())):
        chunks = chunk(para, 100, SEM)
    texts = [c["text"] for c in chunks]
    assert len(texts) == 2
    assert not any("ALPHA" in t and "BETA" in t for t in texts)   # boundary preserved


def test_hybrid_merges_small_paragraphs_without_embedding():
    text = "tiny a.\n\ntiny b.\n\ntiny c."   # each ~2 tokens < semantic_min → structural
    with patch("pkms.embed._embed_batch", side_effect=AssertionError("must not embed")):
        out = chunk(text, 100, SEM)
    assert [c["text"] for c in out] == [c["text"] for c in ing._chunk_fixed(text, 100)]


def test_hybrid_wide_and_small_paragraphs_coexist():
    wide = "ALPHA one. ALPHA two. ALPHA three. BETA four. BETA five. BETA six."
    text = f"tiny intro.\n\n{wide}"
    with patch("pkms.embed._embed_batch", side_effect=_batch(_fake_embed_two_topics())):
        texts = [c["text"] for c in chunk(text, 100, SEM)]
    assert any("tiny intro" in t for t in texts)          # small stays merged/own chunk
    assert any("ALPHA" in t and "BETA" not in t for t in texts)
    assert any("BETA" in t and "ALPHA" not in t for t in texts)
