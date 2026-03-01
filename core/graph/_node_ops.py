"""Node CRUD операции для GraphStorage (mixin)."""

from __future__ import annotations

import json
import logging
import sqlite3
from datetime import datetime, timezone

import aiosqlite

from core.graph.model import Edge, Node, ensure_metadata_defaults

logger = logging.getLogger(__name__)


class NodeOpsMixin:
    """Операции с узлами: upsert, find, soft-delete, merge, retention."""

    async def upsert_node(self, node: Node) -> Node:
        await self._ensure_initialized()

        node_metadata = ensure_metadata_defaults(dict(node.metadata))
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

    async def upsert_nodes_batch(self, nodes_data: list[tuple[Node, dict]]) -> list[Node]:
        """Атомарный upsert списка узлов в одной транзакции."""
        await self._ensure_initialized()
        conn = await self._get_conn()
        saved: list[Node] = []

        await conn.execute("BEGIN")
        try:
            for node, node_metadata in nodes_data:
                if node.key:
                    cursor = await conn.execute(
                        "SELECT id, created_at FROM nodes "
                        "WHERE user_id = ? AND type = ? AND key = ?",
                        (node.user_id, node.type, node.key),
                    )
                    existing = await cursor.fetchone()
                    canonical_id = existing["id"] if existing else node.id
                    created_at = existing["created_at"] if existing else node.created_at
                else:
                    cursor = await conn.execute(
                        "SELECT created_at FROM nodes WHERE id = ?", (node.id,)
                    )
                    existing = await cursor.fetchone()
                    canonical_id = node.id
                    created_at = existing["created_at"] if existing else node.created_at

                await conn.execute(
                    """
                    INSERT OR REPLACE INTO nodes
                      (id, user_id, type, name, text, subtype, key, metadata_json, created_at)
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
                    )
                )
            await conn.commit()
        except Exception:
            await conn.rollback()
            logger.exception("apply_changes_atomic transaction failed, rolled back")
            raise

        return saved

    async def get_node(self, node_id: str) -> Node:
        await self._ensure_initialized()
        conn = await self._get_conn()
        cursor = await conn.execute("SELECT * FROM nodes WHERE id = ?", (node_id,))
        row = await cursor.fetchone()
        if row is None:
            raise KeyError(f"Node not found: {node_id}")
        return _row_to_node(row)

    async def find_nodes(
        self,
        user_id: str,
        node_type: str | None = None,
        name: str | None = None,
        limit: int = 500,
    ) -> list[Node]:
        await self._ensure_initialized()
        query = "SELECT * FROM nodes WHERE user_id = ? AND (is_deleted IS NULL OR is_deleted = 0)"
        params: list[object] = [user_id]
        if node_type:
            query += " AND type = ?"
            params.append(node_type)
        if name:
            query += " AND name = ?"
            params.append(name)
        query += " ORDER BY created_at LIMIT ?"
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
        """Возвращает limit последних узлов по created_at DESC."""
        await self._ensure_initialized()
        conn = await self._get_conn()
        cursor = await conn.execute(
            """
            SELECT * FROM nodes
            WHERE user_id = ? AND type = ?
              AND (is_deleted IS NULL OR is_deleted = 0)
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

    async def count_nodes(self, user_id: str) -> int:
        """Общее количество узлов пользователя."""
        await self._ensure_initialized()
        conn = await self._get_conn()
        cursor = await conn.execute(
            "SELECT COUNT(*) FROM nodes WHERE user_id = ?", (user_id,)
        )
        row = await cursor.fetchone()
        return int(row[0]) if row else 0

    async def soft_delete_node(self, node_id: str) -> None:
        """Пометить узел как удалённый без физического удаления."""
        await self._ensure_initialized()
        conn = await self._get_conn()
        await conn.execute("UPDATE nodes SET is_deleted = 1 WHERE id = ?", (node_id,))
        await conn.commit()

    async def merge_nodes(
        self,
        user_id: str,
        source_node_ids: list[str],
        target_node: Node,
    ) -> Node:
        """Слить несколько узлов в *target_node*.

        * Upsert *target_node*.
        * Перенаправить рёбра *source_node_ids* на *target_node*.
        * Удалить self-loops.
        * Soft-delete исходных узлов.
        """
        if not source_node_ids:
            return await self.upsert_node(target_node)

        await self._ensure_initialized()
        conn = await self._get_conn()

        saved = await self.upsert_node(target_node)

        source_set = list(dict.fromkeys(source_node_ids))
        placeholders = ",".join("?" * len(source_set))

        # Re-point edges: source_node_id → target
        await conn.execute(
            f"UPDATE edges SET source_node_id = ? "
            f"WHERE user_id = ? AND source_node_id IN ({placeholders})",
            [saved.id, user_id, *source_set],
        )
        # Re-point edges: target_node_id → target
        await conn.execute(
            f"UPDATE edges SET target_node_id = ? "
            f"WHERE user_id = ? AND target_node_id IN ({placeholders})",
            [saved.id, user_id, *source_set],
        )

        # Remove self-loops that may have been created
        await conn.execute(
            "DELETE FROM edges WHERE source_node_id = ? AND target_node_id = ?",
            (saved.id, saved.id),
        )

        # Soft-delete source nodes
        await conn.execute(
            f"UPDATE nodes SET is_deleted = 1 "
            f"WHERE user_id = ? AND id IN ({placeholders})",
            [user_id, *source_set],
        )

        await conn.commit()
        return saved

    async def get_nodes_by_retention(
        self,
        user_id: str,
        max_retention: float = 0.3,
        node_types: list[str] | None = None,
        limit: int = 200,
    ) -> list[Node]:
        """Узлы с salience_score ≤ max_retention — кандидаты на забывание."""
        await self._ensure_initialized()
        conn = await self._get_conn()

        query = (
            "SELECT * FROM nodes WHERE user_id = ? AND "
            "(is_deleted IS NULL OR is_deleted = 0)"
        )
        params: list[object] = [user_id]
        if node_types:
            placeholders = ",".join("?" * len(node_types))
            query += f" AND type IN ({placeholders})"
            params.extend(node_types)
        query += " ORDER BY created_at LIMIT ?"
        params.append(limit)

        cursor = await conn.execute(query, params)
        rows = await cursor.fetchall()

        results: list[Node] = []
        for row in rows:
            node = _row_to_node(row)
            salience = float(node.metadata.get("salience_score", 1.0))
            if salience <= max_retention:
                results.append(node)
        return results


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
