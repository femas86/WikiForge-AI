"""Pluggable agentic-memory providers for the Querier.

The Querier's long-term memory is selected via `memory.provider` in
pkms.config.yaml, so the backend is pluggable. Each provider implements the same
recall/store contract; recall returns plain strings injected into the synthesis
prompt — a deliberately uniform, lowest-common-denominator interface so any
memory backend can be swapped in without touching the Querier.

Wave 1 providers (light / local / cheap): none (baseline), mem0 (managed
fact-extraction), amem (local Zettelkasten). MemPalace (spatial / zero-LLM-write)
is planned for Wave 1 too but pending API verification. Wave 2 (graph/temporal +
memory-as-OS: Zep, Cognee, Letta) is deferred — see docs/agentic_memory_eval_design.md.

Default is **none** (a no-op baseline). Providers lazy-import their backends so
unused SDKs are never required.
"""

from __future__ import annotations

import importlib.util
import logging
import os
from typing import Any, Protocol, runtime_checkable

logger = logging.getLogger(__name__)


def _module_available(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


@runtime_checkable
class MemoryProvider(Protocol):
    name: str
    # False when the provider's preconditions (API key, SDK) failed at
    # construction: recall/store become fast no-ops. Validated ONCE, loudly —
    # a misconfigured provider must not masquerade as transient per-query
    # warnings while a whole session runs with memory silently off.
    healthy: bool

    def recall(self, question: str, user_id: str, config: dict[str, Any]) -> list[str]:
        """Return prior-context strings relevant to `question` for `user_id`."""
        ...

    def store(
        self,
        question: str,
        answer_md: str,
        sources: list[str],
        user_id: str,
        session_id: str,
        config: dict[str, Any],
        project: str = "default",
    ) -> None:
        """Persist a Q&A interaction. Must be non-fatal."""
        ...


class NoneProvider:
    """Baseline: no memory. Recall is always empty, store is a no-op."""

    name = "none"
    healthy = True

    def recall(self, question, user_id, config):
        return []

    def store(self, question, answer_md, sources, user_id, session_id, config, project="default"):
        return None


class Mem0Provider:
    """Managed Mem0 platform. Delegates to the (already tested) helpers in
    pkms.querier; imported lazily to avoid an import cycle."""

    name = "mem0"

    def __init__(self):
        self.healthy = True
        if not os.environ.get("MEM0_API_KEY"):
            logger.error("Memory provider 'mem0' misconfigured: MEM0_API_KEY is not set "
                         "— memory is OFF (recall/store are no-ops)")
            self.healthy = False
        elif not _module_available("mem0"):
            logger.error("Memory provider 'mem0' misconfigured: the 'mem0' SDK is not "
                         "installed — memory is OFF (recall/store are no-ops)")
            self.healthy = False

    def recall(self, question, user_id, config):
        if not self.healthy:
            return []
        from pkms.querier import _mem0_recall
        return _mem0_recall(question, user_id, config)

    def store(self, question, answer_md, sources, user_id, session_id, config, project="default"):
        if not self.healthy:
            return None
        from pkms.querier import _mem0_store
        _mem0_store(question, answer_md, sources, user_id, session_id, config, project=project)


class AMemProvider:
    """A-MEM: Zettelkasten / note-linking paradigm (local SQLite implementation).

    Each interaction is a "zettel" (note + embedding); on store, links are formed
    to existing notes above a similarity threshold; on recall, the best-matching
    note plus its 1-hop linked neighbours are returned.
    """

    name = "amem"

    def __init__(self):
        self.healthy = True
        if not _module_available("numpy"):
            logger.error("Memory provider 'amem' misconfigured: numpy is not installed "
                         "— memory is OFF (recall/store are no-ops)")
            self.healthy = False

    def _get_db(self):
        import os
        import sqlite3
        db_dir = os.path.expanduser("~/.pkms")
        os.makedirs(db_dir, exist_ok=True)
        db_path = os.path.join(db_dir, "amem.db")
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("""
            CREATE TABLE IF NOT EXISTS zettels (
                id TEXT PRIMARY KEY,
                content TEXT NOT NULL,
                embedding BLOB NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS links (
                source_id TEXT,
                target_id TEXT,
                PRIMARY KEY (source_id, target_id),
                FOREIGN KEY (source_id) REFERENCES zettels(id) ON DELETE CASCADE,
                FOREIGN KEY (target_id) REFERENCES zettels(id) ON DELETE CASCADE
            )
        """)
        conn.commit()
        return conn

    def recall(self, question, user_id, config):
        if not self.healthy:
            return []
        try:
            import numpy as np
            from pkms.embed import embed

            q_emb = embed(question, config)
            conn = self._get_db()
            cursor = conn.cursor()
            cursor.execute("SELECT id, content, embedding FROM zettels")
            rows = cursor.fetchall()
            if not rows:
                return []

            similarities = []
            for row in rows:
                emb = np.frombuffer(row["embedding"], dtype=np.float32)
                sim = np.dot(q_emb, emb) / (np.linalg.norm(q_emb) * np.linalg.norm(emb))
                similarities.append((row["id"], row["content"], sim))

            similarities.sort(key=lambda x: x[2], reverse=True)
            if not similarities or similarities[0][2] < 0.6:
                return []

            best_id, best_content, _ = similarities[0]
            context = [best_content]

            # 1-hop connected neighbours (bidirectional links)
            cursor.execute("""
                SELECT content FROM zettels WHERE id IN (
                    SELECT target_id FROM links WHERE source_id = ?
                    UNION
                    SELECT source_id FROM links WHERE target_id = ?
                )
            """, (best_id, best_id))
            for link_row in cursor.fetchall():
                context.append(link_row["content"])
            return context
        except Exception as exc:
            logger.warning("AMem recall failed (non-fatal): %s", exc)
            return []

    def store(self, question, answer_md, sources, user_id, session_id, config, project="default"):
        if not self.healthy:
            return None
        try:
            import uuid
            import numpy as np
            from pkms.embed import embed

            content = f"User {user_id} Q&A:\nQuestion: {question}\nAnswer: {answer_md}"
            new_emb = embed(content, config)
            new_id = str(uuid.uuid4())
            new_emb_bytes = np.array(new_emb, dtype=np.float32).tobytes()

            conn = self._get_db()
            cursor = conn.cursor()
            cursor.execute("SELECT id, embedding FROM zettels")
            rows = cursor.fetchall()

            cursor.execute(
                "INSERT INTO zettels (id, content, embedding) VALUES (?, ?, ?)",
                (new_id, content, new_emb_bytes),
            )

            for row in rows:
                emb = np.frombuffer(row["embedding"], dtype=np.float32)
                sim = np.dot(new_emb, emb) / (np.linalg.norm(new_emb) * np.linalg.norm(emb))
                if sim >= 0.75:
                    cursor.execute(
                        "INSERT OR IGNORE INTO links (source_id, target_id) VALUES (?, ?)",
                        (new_id, row["id"]),
                    )
                    cursor.execute(
                        "INSERT OR IGNORE INTO links (source_id, target_id) VALUES (?, ?)",
                        (row["id"], new_id),
                    )
            conn.commit()
        except Exception as exc:
            logger.warning("AMem store failed (non-fatal): %s", exc)


class MemPalaceProvider:
    """MemPalace: spatial / method-of-loci, verbatim **zero-LLM-write** memory.

    Each interaction is stored as a "drawer" (wing=user_id, room="qa"); recall is
    hybrid vector+BM25 search scoped to the user's wing. Uses MemPalace's own
    ChromaDB embedder (all-MiniLM-L6-v2). Heavy optional dependency — install with
    `pip install '.[mempalace]'`. See docs/mempalace_sdk.md.
    """

    name = "mempalace"

    def __init__(self):
        self.healthy = True
        if not _module_available("mempalace"):
            logger.error("Memory provider 'mempalace' misconfigured: the 'mempalace' "
                         "package is not installed (pip install '.[mempalace]') "
                         "— memory is OFF (recall/store are no-ops)")
            self.healthy = False

    def _palace_path(self, config):
        import os
        p = (config.get("mempalace") or {}).get("palace_path") or os.path.expanduser("~/.pkms/mempalace")
        os.makedirs(p, exist_ok=True)
        return p

    def recall(self, question, user_id, config):
        if not self.healthy:
            return []
        try:
            from mempalace import searcher
            top_k = (config.get("query") or {}).get("mem0_recall", 5)
            res = searcher.search_memories(question, self._palace_path(config), wing=user_id, n_results=top_k)
            results = res.get("results", []) if isinstance(res, dict) else []
            return [r.get("text", "") for r in results if r.get("text")]
        except Exception as exc:
            logger.warning("MemPalace recall failed (non-fatal): %s", exc)
            return []

    def store(self, question, answer_md, sources, user_id, session_id, config, project="default"):
        if not self.healthy:
            return None
        try:
            import uuid
            from mempalace import miner
            col = miner.get_collection(self._palace_path(config), create=True)
            content = f"Question: {question}\nAnswer: {answer_md}"
            # unique source_file avoids drawer-id collisions (id = hash(wing,room,source_file,chunk_index))
            miner.add_drawer(col, wing=user_id, room="qa", content=content,
                             source_file=f"{session_id}:{uuid.uuid4().hex[:8]}", chunk_index=0, agent="pkms")
        except Exception as exc:
            logger.warning("MemPalace store failed (non-fatal): %s", exc)


_PROVIDERS: dict[str, type] = {
    "none": NoneProvider,
    "mem0": Mem0Provider,
    "amem": AMemProvider,
    "mempalace": MemPalaceProvider,
    # Wave 2 (deferred): zep, cognee, letta — see docs/agentic_memory_eval_design.md
}


def get_provider(config: dict[str, Any]) -> MemoryProvider:
    """Build the configured memory provider. Defaults to 'none' (baseline).

    Preconditions are validated in the provider constructor: an unhealthy
    provider logs one loud error there and degrades to no-op recall/store.
    """
    name = (config.get("memory") or {}).get("provider", "none")
    cls = _PROVIDERS.get(name)
    if cls is None:
        logger.warning("Unknown memory provider %r — falling back to 'none'", name)
        cls = NoneProvider
    return cls()
