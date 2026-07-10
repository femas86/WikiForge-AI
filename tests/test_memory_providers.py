import sqlite3
import sys
from unittest.mock import MagicMock, patch

from pkms.memory import AMemProvider, MemPalaceProvider, get_provider


def test_get_amem_provider():
    assert isinstance(get_provider({"memory": {"provider": "amem"}}), AMemProvider)


def test_amem_provider_sqlite_store_and_recall(tmp_path):
    """A-MEM stores a zettel and recalls it (cosine over a local SQLite store)."""
    provider = AMemProvider()
    db_path = str(tmp_path / "amem.db")

    def mock_get_db(self):
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("""
            CREATE TABLE IF NOT EXISTS zettels (
                id TEXT PRIMARY KEY, content TEXT NOT NULL, embedding BLOB NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS links (
                source_id TEXT, target_id TEXT, PRIMARY KEY (source_id, target_id)
            )
        """)
        conn.commit()
        return conn

    # embed now requires (text, config) — patched to ignore args and return a fixed vector
    with patch.object(AMemProvider, "_get_db", mock_get_db), \
         patch("pkms.embed._embed_batch", side_effect=lambda ts, c: [[0.1] * 768] * len(ts)):
        provider.store("Test Q", "Test A", ["src"], "alice", "sess", {})
        recalled = provider.recall("Test Q", "alice", {})
        assert len(recalled) == 1
        assert "Test Q" in recalled[0]


def test_get_mempalace_provider():
    assert isinstance(get_provider({"memory": {"provider": "mempalace"}}), MemPalaceProvider)


def test_mempalace_provider_store_and_recall(tmp_path):
    """MemPalace store→add_drawer (wing=user), recall→parse search_memories results[].text.
    The heavy `mempalace` dep is faked via sys.modules so the test needs no install."""
    fake_miner = MagicMock()
    fake_searcher = MagicMock()
    fake_searcher.search_memories.return_value = {
        "results": [{"text": "Question: Q?\nAnswer: Qdrant."}, {"text": ""}]
    }
    fake_pkg = MagicMock()
    fake_pkg.miner = fake_miner
    fake_pkg.searcher = fake_searcher
    cfg = {"mempalace": {"palace_path": str(tmp_path / "palace")}, "query": {"mem0_recall": 3}}

    with patch.dict(sys.modules, {
        "mempalace": fake_pkg,
        "mempalace.miner": fake_miner,
        "mempalace.searcher": fake_searcher,
    }):
        # find_spec can't see sys.modules MagicMocks (no real __spec__) — fake
        # the availability probe so construction reports healthy.
        with patch("pkms.memory._module_available", return_value=True):
            p = MemPalaceProvider()
        assert p.healthy
        p.store("Q?", "Qdrant.", ["s"], "alice", "sess", cfg)
        recalled = p.recall("vector db?", "alice", cfg)

    assert fake_miner.add_drawer.called
    kw = fake_miner.add_drawer.call_args.kwargs
    assert kw["wing"] == "alice" and kw["room"] == "qa"
    assert "Qdrant." in kw["content"]
    # recall returns results[].text, dropping empties
    assert recalled == ["Question: Q?\nAnswer: Qdrant."]
    _, skw = fake_searcher.search_memories.call_args
    assert skw["wing"] == "alice" and skw["n_results"] == 3
