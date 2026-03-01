from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import aiosqlite

from core.graph.model import Edge, Node, ensure_metadata_defaults
from core.graph._node_ops import NodeOpsMixin, _row_to_node
from core.graph._edge_ops import EdgeOpsMixin, _row_to_edge
from core.graph._mood_ops import MoodOpsMixin
from core.graph._scheduler_ops import SchedulerOpsMixin

logger = logging.getLogger(__name__)


class GraphStorage(NodeOpsMixin, EdgeOpsMixin, MoodOpsMixin, SchedulerOpsMixin):
    """Единая точка доступа к графовому хранилищу.

    Логика операций разнесена по миксинам:
    - NodeOpsMixin     — узлы (upsert, find, merge, soft-delete)
    - EdgeOpsMixin     — рёбра (add, get, list, filter)
    - MoodOpsMixin     — mood_snapshots
    - SchedulerOpsMixin — scheduler_state, signal_feedback, user/activity
    """

    def __init__(self, db_path: str | Path = "data/self_os.db") -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialized = False
        self._init_lock = asyncio.Lock()
        self._conn: aiosqlite.Connection | None = None
        self._conn_lock = asyncio.Lock()

    async def _get_conn(self) -> aiosqlite.Connection:
        if self._conn is None:
            async with self._conn_lock:
                if self._conn is None:
                    self._conn = await aiosqlite.connect(str(self.db_path))
                    self._conn.row_factory = aiosqlite.Row
        return self._conn

    async def close(self) -> None:
        if self._conn is not None:
            await self._conn.close()
            self._conn = None
            self._initialized = False

    async def _ensure_initialized(self) -> None:
        if self._initialized:
            return

        async with self._init_lock:
            if self._initialized:
                return

            conn = await self._get_conn()
            await conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS nodes (
                    id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    type TEXT NOT NULL,
                    name TEXT,
                    text TEXT,
                    subtype TEXT,
                    key TEXT,
                    metadata_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );

                CREATE UNIQUE INDEX IF NOT EXISTS idx_nodes_user_type_key
                    ON nodes(user_id, type, key)
                    WHERE key IS NOT NULL;

                CREATE INDEX IF NOT EXISTS idx_nodes_user_type
                    ON nodes(user_id, type);

                CREATE INDEX IF NOT EXISTS idx_nodes_user_created
                    ON nodes(user_id, created_at DESC);

                CREATE TABLE IF NOT EXISTS edges (
                    id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    source_node_id TEXT NOT NULL,
                    target_node_id TEXT NOT NULL,
                    relation TEXT NOT NULL,
                    metadata_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(source_node_id) REFERENCES nodes(id),
                    FOREIGN KEY(target_node_id) REFERENCES nodes(id)
                );

                CREATE UNIQUE INDEX IF NOT EXISTS idx_edges_unique
                    ON edges(user_id, source_node_id, target_node_id, relation);

                CREATE INDEX IF NOT EXISTS idx_edges_user_created
                    ON edges(user_id, created_at DESC);

                CREATE TABLE IF NOT EXISTS mood_snapshots (
                    id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    valence_avg REAL NOT NULL,
                    arousal_avg REAL NOT NULL,
                    dominance_avg REAL NOT NULL DEFAULT 0.0,
                    intensity_avg REAL NOT NULL DEFAULT 0.5,
                    dominant_label TEXT,
                    sample_count INTEGER NOT NULL DEFAULT 1
                );

                CREATE INDEX IF NOT EXISTS idx_mood_snapshots_user_ts
                    ON mood_snapshots (user_id, timestamp DESC);

                CREATE TABLE IF NOT EXISTS scheduler_state (
                    user_id TEXT PRIMARY KEY,
                    last_proactive_at TEXT,
                    last_checked_at TEXT,
                    total_sent INTEGER NOT NULL DEFAULT 0
                );

                CREATE TABLE IF NOT EXISTS signal_feedback (
                    id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    signal_type TEXT NOT NULL,
                    signal_score REAL NOT NULL,
                    was_helpful INTEGER NOT NULL,
                    sent_at TEXT NOT NULL,
                    feedback_at TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_signal_feedback_user_type
                    ON signal_feedback(user_id, signal_type);
                """
            )
            # ── Sprint-0 migrations (backward-compatible ALTER TABLE) ──
            _migrations = [
                "ALTER TABLE nodes ADD COLUMN is_deleted INTEGER NOT NULL DEFAULT 0",
                # mood_snapshots — fields for future predictive engine
                "ALTER TABLE mood_snapshots ADD COLUMN stressor_tags TEXT DEFAULT '[]'",
                "ALTER TABLE mood_snapshots ADD COLUMN active_parts_keys TEXT DEFAULT '[]'",
                "ALTER TABLE mood_snapshots ADD COLUMN intervention_applied TEXT",
                "ALTER TABLE mood_snapshots ADD COLUMN feedback_score INTEGER",
            ]
            for stmt in _migrations:
                with contextlib.suppress(sqlite3.OperationalError):
                    await conn.execute(stmt)

            # intervention_outcomes — minimal OutcomeTracker table
            await conn.execute(
                """
                CREATE TABLE IF NOT EXISTS intervention_outcomes (
                    id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    intervention_type TEXT NOT NULL,
                    pre_valence REAL,
                    pre_arousal REAL,
                    pre_dominance REAL,
                    post_valence REAL,
                    post_arousal REAL,
                    post_dominance REAL,
                    user_feedback INTEGER,
                    created_at TEXT NOT NULL
                )
                """
            )
            await conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_intervention_outcomes_user
                    ON intervention_outcomes(user_id, created_at DESC)
                """
            )
            await conn.commit()
            self._initialized = True

    # ── Search ─────────────────────────────────────────────────────

    async def hybrid_search(
        self,
        user_id: str,
        query_text: str,
        alpha: float = 0.7,
        top_k: int = 10,
        use_rrf: bool = False,
    ) -> list[tuple[Node, float]]:
        """Hybrid sparse search over user nodes."""
        from core.search.hybrid_search import HybridSearchEngine

        nodes = await self.find_nodes(user_id, limit=500)
        engine = HybridSearchEngine(alpha=alpha)
        return engine.search(
            query_text=query_text,
            query_embedding=None,
            nodes=nodes,
            top_k=top_k,
            use_rrf=use_rrf,
        )


# Backward-compat alias — canonical implementation lives in core.utils.math
from core.utils.math import cosine_similarity as _cosine_similarity  # noqa: E402
