from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
import math
from pathlib import Path
from uuid import uuid4

import aiosqlite

from core.graph.model import Edge, Node


class GraphStorage:
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
            try:
                await conn.execute("ALTER TABLE nodes ADD COLUMN embedding_json TEXT")
            except Exception:
                pass
            await conn.commit()
            self._initialized = True

    async def upsert_node(self, node: Node) -> Node:
        await self._ensure_initialized()

        node_metadata = dict(node.metadata)
        if node.type == "EMOTION" and "created_at" not in node_metadata:
            node_metadata["created_at"] = datetime.now(timezone.utc).isoformat()

        conn = await self._get_conn()
        if node.key:
            cursor = await conn.execute(
                """
                SELECT * FROM nodes
                WHERE user_id = ? AND type = ? AND key = ?
                """,
                (node.user_id, node.type, node.key),
            )
            existing = await cursor.fetchone()
            canonical_id = existing["id"] if existing else node.id
            created_at = existing["created_at"] if existing else node.created_at
            embedding_json = (
                json.dumps(node.embedding)
                if node.embedding is not None
                else (existing["embedding_json"] if existing else None)
            )

            await conn.execute(
                """
                INSERT OR REPLACE INTO nodes (id, user_id, type, name, text, subtype, key, metadata_json, created_at, embedding_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    canonical_id,
                    node.user_id,
                    node.type,
                    node.name,
                    node.text,
                    node.subtype,
                    node.key,
                    json.dumps(node_metadata, ensure_ascii=False),
                    created_at,
                    embedding_json,
                ),
            )
            await conn.commit()
            return Node(
                id=canonical_id,
                user_id=node.user_id,
                type=node.type,
                name=node.name,
                text=node.text,
                subtype=node.subtype,
                key=node.key,
                metadata=node_metadata,
                created_at=created_at,
                embedding=node.embedding if node.embedding is not None else (json.loads(embedding_json) if embedding_json else None),
            )

        cursor = await conn.execute("SELECT created_at, embedding_json FROM nodes WHERE id = ?", (node.id,))
        existing = await cursor.fetchone()
        created_at = existing["created_at"] if existing else node.created_at
        embedding_json = (
            json.dumps(node.embedding)
            if node.embedding is not None
            else (existing["embedding_json"] if existing else None)
        )

        await conn.execute(
            """
            INSERT OR REPLACE INTO nodes (id, user_id, type, name, text, subtype, key, metadata_json, created_at, embedding_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                node.id,
                node.user_id,
                node.type,
                node.name,
                node.text,
                node.subtype,
                None,
                json.dumps(node_metadata, ensure_ascii=False),
                created_at,
                embedding_json,
            ),
        )
        await conn.commit()
        return Node(
            id=node.id,
            user_id=node.user_id,
            type=node.type,
            name=node.name,
            text=node.text,
            subtype=node.subtype,
            key=None,
            metadata=node_metadata,
            created_at=created_at,
            embedding=node.embedding if node.embedding is not None else (json.loads(embedding_json) if embedding_json else None),
        )

    async def upsert_nodes_batch(self, nodes_data: list[tuple[Node, dict]]) -> list[Node]:
        """
        Атомарный апсерт списка узлов в одной транзакции.
        nodes_data: [(node, metadata_dict), ...]
        Возвращает сохранённые узлы в том же порядке.
        """
        await self._ensure_initialized()
        conn = await self._get_conn()
        saved: list[Node] = []

        await conn.execute("BEGIN")
        try:
            for node, node_metadata in nodes_data:
                if node.key:
                    cursor = await conn.execute(
                        "SELECT id, created_at, embedding_json FROM nodes "
                        "WHERE user_id = ? AND type = ? AND key = ?",
                        (node.user_id, node.type, node.key),
                    )
                    existing = await cursor.fetchone()
                    canonical_id = existing["id"] if existing else node.id
                    created_at = existing["created_at"] if existing else node.created_at
                else:
                    cursor = await conn.execute(
                        "SELECT created_at, embedding_json FROM nodes WHERE id = ?", (node.id,)
                    )
                    existing = await cursor.fetchone()
                    canonical_id = node.id
                    created_at = existing["created_at"] if existing else node.created_at
                embedding_json = (
                    json.dumps(node.embedding)
                    if node.embedding is not None
                    else (existing["embedding_json"] if existing else None)
                )

                await conn.execute(
                    """
                    INSERT OR REPLACE INTO nodes
                      (id, user_id, type, name, text, subtype, key, metadata_json, created_at, embedding_json)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        canonical_id,
                        node.user_id,
                        node.type,
                        node.name,
                        node.text,
                        node.subtype,
                        node.key,
                        json.dumps(node_metadata, ensure_ascii=False),
                        created_at,
                        embedding_json,
                    ),
                )
                saved.append(
                    Node(
                        id=canonical_id,
                        user_id=node.user_id,
                        type=node.type,
                        name=node.name,
                        text=node.text,
                        subtype=node.subtype,
                        key=node.key,
                        metadata=node_metadata,
                        created_at=created_at,
                        embedding=node.embedding if node.embedding is not None else (json.loads(embedding_json) if embedding_json else None),
                    )
                )
            await conn.commit()
        except Exception:
            await conn.rollback()
            raise

        return saved

    async def save_mood_snapshot(self, snapshot: dict) -> None:
        await self._ensure_initialized()

        conn = await self._get_conn()
        await conn.execute(
            """
            INSERT OR REPLACE INTO mood_snapshots (
                id, user_id, timestamp, valence_avg, arousal_avg,
                dominance_avg, intensity_avg, dominant_label, sample_count
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                snapshot["id"],
                snapshot["user_id"],
                snapshot["timestamp"],
                snapshot["valence_avg"],
                snapshot["arousal_avg"],
                snapshot.get("dominance_avg", 0.0),
                snapshot.get("intensity_avg", 0.5),
                snapshot.get("dominant_label"),
                snapshot.get("sample_count", 1),
            ),
        )
        await conn.execute(
            """
            DELETE FROM mood_snapshots
            WHERE user_id = ?
              AND id NOT IN (
                  SELECT id FROM mood_snapshots
                  WHERE user_id = ?
                  ORDER BY timestamp DESC
                  LIMIT 30
              )
            """,
            (snapshot["user_id"], snapshot["user_id"]),
        )
        await conn.commit()

    async def get_latest_mood_snapshot(self, user_id: str) -> dict | None:
        await self._ensure_initialized()

        conn = await self._get_conn()
        cursor = await conn.execute(
            """
            SELECT * FROM mood_snapshots
            WHERE user_id = ?
            ORDER BY timestamp DESC
            LIMIT 1
            """,
            (user_id,),
        )
        row = await cursor.fetchone()
        return dict(row) if row else None

    async def get_mood_snapshots(self, user_id: str, limit: int = 5) -> list[dict]:
        await self._ensure_initialized()

        conn = await self._get_conn()
        cursor = await conn.execute(
            """
            SELECT * FROM mood_snapshots
            WHERE user_id = ?
            ORDER BY timestamp DESC
            LIMIT ?
            """,
            (user_id, limit),
        )
        rows = await cursor.fetchall()
        return [dict(row) for row in rows]

    async def add_edge(self, edge: Edge) -> Edge:
        await self._ensure_initialized()

        conn = await self._get_conn()
        cursor = await conn.execute(
            """
            SELECT id FROM edges
            WHERE user_id = ? AND source_node_id = ? AND target_node_id = ? AND relation = ?
            """,
            (edge.user_id, edge.source_node_id, edge.target_node_id, edge.relation),
        )
        existing = await cursor.fetchone()
        if existing:
            return Edge(
                id=existing["id"],
                user_id=edge.user_id,
                source_node_id=edge.source_node_id,
                target_node_id=edge.target_node_id,
                relation=edge.relation,
                metadata=edge.metadata,
            )

        await conn.execute(
            """
            INSERT INTO edges (id, user_id, source_node_id, target_node_id, relation, metadata_json, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                edge.id,
                edge.user_id,
                edge.source_node_id,
                edge.target_node_id,
                edge.relation,
                json.dumps(edge.metadata, ensure_ascii=False),
                edge.created_at,
            ),
        )
        await conn.commit()
        return edge

    async def get_node(self, node_id: str) -> Node:
        await self._ensure_initialized()

        conn = await self._get_conn()
        cursor = await conn.execute("SELECT * FROM nodes WHERE id = ?", (node_id,))
        row = await cursor.fetchone()
        if row is None:
            raise KeyError(f"Node not found: {node_id}")
        return _row_to_node(row)

    async def get_edge(self, edge_id: str) -> Edge:
        await self._ensure_initialized()

        conn = await self._get_conn()
        cursor = await conn.execute("SELECT * FROM edges WHERE id = ?", (edge_id,))
        row = await cursor.fetchone()
        if row is None:
            raise KeyError(f"Edge not found: {edge_id}")
        return _row_to_edge(row)

    async def find_nodes(
        self,
        user_id: str,
        node_type: str | None = None,
        name: str | None = None,
        limit: int = 500,
    ) -> list[Node]:
        await self._ensure_initialized()

        query = "SELECT * FROM nodes WHERE user_id = ?"
        params: list[object] = [user_id]
        if node_type:
            query += " AND type = ?"
            params.append(node_type)
        if name:
            query += " AND name = ?"
            params.append(name)
        query += " ORDER BY created_at"
        query += " LIMIT ?"
        params.append(limit)

        conn = await self._get_conn()
        cursor = await conn.execute(query, params)
        rows = await cursor.fetchall()
        return [_row_to_node(row) for row in rows]

    async def find_nodes_recent(
        self,
        user_id: str,
        node_type: str,
        limit: int = 5,
    ) -> list[Node]:
        """Возвращает limit последних узлов по created_at DESC — через SQL, без Python sort."""
        await self._ensure_initialized()
        conn = await self._get_conn()
        cursor = await conn.execute(
            """
            SELECT * FROM nodes
            WHERE user_id = ? AND type = ?
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (user_id, node_type, limit),
        )
        rows = await cursor.fetchall()
        return [_row_to_node(row) for row in rows]

    async def find_by_key(self, user_id: str, node_type: str, key: str) -> Node | None:
        await self._ensure_initialized()

        conn = await self._get_conn()
        cursor = await conn.execute(
            "SELECT * FROM nodes WHERE user_id = ? AND type = ? AND key = ?",
            (user_id, node_type, key),
        )
        row = await cursor.fetchone()
        return _row_to_node(row) if row else None

    async def get_nodes_by_ids(self, user_id: str, node_ids: list[str]) -> list[Node]:
        """Возвращает узлы пользователя по списку id одним SQL-запросом."""
        if not node_ids:
            return []

        await self._ensure_initialized()
        conn = await self._get_conn()

        unique_ids = list(dict.fromkeys(node_ids))
        placeholders = ", ".join("?" for _ in unique_ids)
        query = f"SELECT * FROM nodes WHERE user_id = ? AND id IN ({placeholders})"
        cursor = await conn.execute(query, [user_id, *unique_ids])
        rows = await cursor.fetchall()
        return [_row_to_node(row) for row in rows]

    async def list_edges(self, user_id: str) -> list[Edge]:
        await self._ensure_initialized()

        conn = await self._get_conn()
        cursor = await conn.execute(
            "SELECT * FROM edges WHERE user_id = ? ORDER BY created_at",
            (user_id,),
        )
        rows = await cursor.fetchall()
        return [_row_to_edge(row) for row in rows]

    async def get_edges_by_relation(self, user_id: str, relation: str) -> list[Edge]:
        """Все рёбра пользователя с указанным relation. Быстрее чем list_edges + filter."""
        # NOTE: added storage method get_edges_by_relation for performance.
        await self._ensure_initialized()
        conn = await self._get_conn()
        cursor = await conn.execute(
            "SELECT * FROM edges WHERE user_id = ? AND relation = ? ORDER BY created_at",
            (user_id, relation),
        )
        rows = await cursor.fetchall()
        return [_row_to_edge(row) for row in rows]

    async def get_edges_to_node(self, user_id: str, target_node_id: str) -> list[Edge]:
        """Все рёбра входящие в указанный узел."""
        await self._ensure_initialized()
        conn = await self._get_conn()
        cursor = await conn.execute(
            "SELECT * FROM edges WHERE user_id = ? AND target_node_id = ?",
            (user_id, target_node_id),
        )
        rows = await cursor.fetchall()
        return [_row_to_edge(row) for row in rows]

    async def get_edges_from_node(self, user_id: str, source_node_id: str) -> list[Edge]:
        """Все рёбра исходящие из указанного узла."""
        await self._ensure_initialized()
        conn = await self._get_conn()
        cursor = await conn.execute(
            "SELECT * FROM edges WHERE user_id = ? AND source_node_id = ?",
            (user_id, source_node_id),
        )
        rows = await cursor.fetchall()
        return [_row_to_edge(row) for row in rows]

    async def count_nodes(self, user_id: str) -> int:
        """Общее количество узлов пользователя."""
        await self._ensure_initialized()
        conn = await self._get_conn()
        cursor = await conn.execute(
            "SELECT COUNT(*) FROM nodes WHERE user_id = ?", (user_id,)
        )
        row = await cursor.fetchone()
        return int(row[0]) if row else 0

    async def get_all_user_ids(self) -> list[str]:
        """Все уникальные user_id у которых есть узлы в графе."""
        await self._ensure_initialized()
        conn = await self._get_conn()
        cursor = await conn.execute(
            "SELECT DISTINCT user_id FROM nodes ORDER BY user_id"
        )
        rows = await cursor.fetchall()
        return [row[0] for row in rows]

    async def get_last_activity_at(self, user_id: str) -> str | None:
        """
        ISO datetime последнего созданного узла пользователя.
        Используется чтобы не беспокоить неактивных пользователей.
        """
        await self._ensure_initialized()
        conn = await self._get_conn()
        cursor = await conn.execute(
            "SELECT MAX(created_at) FROM nodes WHERE user_id = ?",
            (user_id,),
        )
        row = await cursor.fetchone()
        return row[0] if row and row[0] else None

    async def get_scheduler_state(self, user_id: str) -> dict | None:
        """Состояние scheduler для пользователя."""
        await self._ensure_initialized()
        conn = await self._get_conn()
        cursor = await conn.execute(
            "SELECT * FROM scheduler_state WHERE user_id = ?",
            (user_id,),
        )
        row = await cursor.fetchone()
        return dict(row) if row else None

    async def upsert_scheduler_state(
        self,
        user_id: str,
        last_proactive_at: str | None = None,
        last_checked_at: str | None = None,
        increment_sent: bool = False,
    ) -> None:
        """Обновляет состояние scheduler. Создаёт запись если нет."""
        await self._ensure_initialized()
        conn = await self._get_conn()
        now = datetime.now(timezone.utc).isoformat()

        existing = await self.get_scheduler_state(user_id)
        if existing is None:
            await conn.execute(
                """
                INSERT INTO scheduler_state
                  (user_id, last_proactive_at, last_checked_at, total_sent)
                VALUES (?, ?, ?, ?)
                """,
                (
                    user_id,
                    last_proactive_at,
                    last_checked_at or now,
                    1 if increment_sent else 0,
                ),
            )
        else:
            total = existing.get("total_sent", 0) + (1 if increment_sent else 0)
            await conn.execute(
                """
                UPDATE scheduler_state
                SET last_proactive_at = COALESCE(?, last_proactive_at),
                    last_checked_at   = COALESCE(?, last_checked_at),
                    total_sent        = ?
                WHERE user_id = ?
                """,
                (last_proactive_at, last_checked_at or now, total, user_id),
            )
        await conn.commit()

    async def save_signal_feedback(
        self,
        user_id: str,
        signal_type: str,
        signal_score: float,
        was_helpful: bool,
        sent_at: str,
    ) -> None:
        await self._ensure_initialized()
        conn = await self._get_conn()
        await conn.execute(
            """
            INSERT INTO signal_feedback (
                id, user_id, signal_type, signal_score, was_helpful, sent_at, feedback_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                str(uuid4()),
                user_id,
                signal_type,
                float(signal_score),
                1 if was_helpful else 0,
                sent_at,
                datetime.now(timezone.utc).isoformat(),
            ),
        )
        await conn.commit()

    async def get_signal_feedback(
        self,
        user_id: str,
        signal_type: str | None = None,
        limit: int = 100,
    ) -> list[dict]:
        await self._ensure_initialized()
        conn = await self._get_conn()
        if signal_type:
            cursor = await conn.execute(
                """
                SELECT * FROM signal_feedback
                WHERE user_id = ? AND signal_type = ?
                ORDER BY feedback_at DESC
                LIMIT ?
                """,
                (user_id, signal_type, limit),
            )
        else:
            cursor = await conn.execute(
                """
                SELECT * FROM signal_feedback
                WHERE user_id = ?
                ORDER BY feedback_at DESC
                LIMIT ?
                """,
                (user_id, limit),
            )
        rows = await cursor.fetchall()
        return [
            {
                **dict(row),
                "was_helpful": bool(row["was_helpful"]),
            }
            for row in rows
        ]

    async def save_node_embedding(self, node_id: str, embedding: list[float]) -> None:
        """Сохраняет embedding отдельно, не трогая остальные поля."""
        await self._ensure_initialized()
        conn = await self._get_conn()
        await conn.execute(
            "UPDATE nodes SET embedding_json = ? WHERE id = ?",
            (json.dumps(embedding), node_id),
        )
        await conn.commit()

    async def hybrid_search(
        self,
        user_id: str,
        query_text: str,
        query_embedding: list[float] | None = None,
        alpha: float = 0.7,
        top_k: int = 10,
        use_rrf: bool = False,
    ) -> list[tuple[Node, float]]:
        """Hybrid dense + sparse search over user nodes.

        Parameters
        ----------
        user_id:
            Owner of the nodes to search.
        query_text:
            Raw query string used for TF-IDF sparse scoring.
        query_embedding:
            Optional dense query vector for cosine similarity scoring.
        alpha:
            Weight for dense scores (0 = sparse only, 1 = dense only).
        top_k:
            Maximum number of results to return.
        use_rrf:
            If ``True``, use Reciprocal Rank Fusion instead of weighted sum.
        """
        from core.search.hybrid_search import HybridSearchEngine

        nodes = await self.find_nodes(user_id, limit=500)
        engine = HybridSearchEngine(alpha=alpha)
        return engine.search(
            query_text=query_text,
            query_embedding=query_embedding,
            nodes=nodes,
            top_k=top_k,
            use_rrf=use_rrf,
        )

    async def find_similar_nodes(
        self,
        user_id: str,
        query_embedding: list[float],
        top_k: int = 5,
        node_types: list[str] | None = None,
        min_similarity: float = 0.75,
    ) -> list[tuple[Node, float]]:
        """
        Косинусное сходство без NumPy — через stdlib math.
        Возвращает [(node, similarity_score), ...] отсортированный DESC.
        Загружает только узлы у которых есть embedding_json.
        """
        await self._ensure_initialized()
        conn = await self._get_conn()

        query = "SELECT * FROM nodes WHERE user_id = ? AND embedding_json IS NOT NULL"
        params: list[object] = [user_id]
        if node_types:
            placeholders = ",".join("?" * len(node_types))
            query += f" AND type IN ({placeholders})"
            params.extend(node_types)

        cursor = await conn.execute(query, params)
        rows = await cursor.fetchall()

        results: list[tuple[Node, float]] = []
        for row in rows:
            node = _row_to_node(row)
            if node.embedding is None:
                continue
            sim = _cosine_similarity(query_embedding, node.embedding)
            if sim >= min_similarity:
                results.append((node, sim))

        results.sort(key=lambda item: item[1], reverse=True)
        return results[:top_k]


def _row_to_node(row: aiosqlite.Row) -> Node:
    embedding_raw = row["embedding_json"] if "embedding_json" in row.keys() else None
    embedding = json.loads(embedding_raw) if embedding_raw else None
    return Node(
        id=row["id"],
        user_id=row["user_id"],
        type=row["type"],
        name=row["name"],
        text=row["text"],
        subtype=row["subtype"],
        key=row["key"],
        metadata=json.loads(row["metadata_json"]),
        created_at=row["created_at"],
        embedding=embedding,
    )


def _row_to_edge(row: aiosqlite.Row) -> Edge:
    return Edge(
        id=row["id"],
        user_id=row["user_id"],
        source_node_id=row["source_node_id"],
        target_node_id=row["target_node_id"],
        relation=row["relation"],
        metadata=json.loads(row["metadata_json"]),
        created_at=row["created_at"],
    )


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Stdlib-only cosine similarity."""
    if len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)
