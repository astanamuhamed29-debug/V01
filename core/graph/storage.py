from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path

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
                """
            )
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

            await conn.execute(
                """
                INSERT OR REPLACE INTO nodes (id, user_id, type, name, text, subtype, key, metadata_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
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
            )

        cursor = await conn.execute("SELECT created_at FROM nodes WHERE id = ?", (node.id,))
        existing = await cursor.fetchone()
        created_at = existing["created_at"] if existing else node.created_at

        await conn.execute(
            """
            INSERT OR REPLACE INTO nodes (id, user_id, type, name, text, subtype, key, metadata_json, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
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
        )

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

    async def list_edges(self, user_id: str) -> list[Edge]:
        await self._ensure_initialized()

        conn = await self._get_conn()
        cursor = await conn.execute(
            "SELECT * FROM edges WHERE user_id = ? ORDER BY created_at",
            (user_id,),
        )
        rows = await cursor.fetchall()
        return [_row_to_edge(row) for row in rows]


def _row_to_node(row: aiosqlite.Row) -> Node:
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
