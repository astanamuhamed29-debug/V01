"""Edge CRUD операции для GraphStorage (mixin)."""

from __future__ import annotations

import json
from typing import Protocol

import aiosqlite

from core.graph.model import Edge


class _GraphStorageLike(Protocol):
    async def _ensure_initialized(self) -> None: ...
    async def _get_conn(self) -> aiosqlite.Connection: ...


class EdgeOpsMixin:
    """Операции с рёбрами: add, get, list, filter."""

    async def add_edge(self: _GraphStorageLike, edge: Edge) -> Edge:
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

    async def get_edge(self: _GraphStorageLike, edge_id: str) -> Edge:
        await self._ensure_initialized()
        conn = await self._get_conn()
        cursor = await conn.execute("SELECT * FROM edges WHERE id = ?", (edge_id,))
        row = await cursor.fetchone()
        if row is None:
            raise KeyError(f"Edge not found: {edge_id}")
        return _row_to_edge(row)

    async def list_edges(self: _GraphStorageLike, user_id: str) -> list[Edge]:
        await self._ensure_initialized()
        conn = await self._get_conn()
        cursor = await conn.execute(
            "SELECT * FROM edges WHERE user_id = ? ORDER BY created_at",
            (user_id,),
        )
        rows = await cursor.fetchall()
        return [_row_to_edge(row) for row in rows]

    async def get_edges_by_relation(self: _GraphStorageLike, user_id: str, relation: str) -> list[Edge]:
        """Все рёбра пользователя с указанным relation."""
        await self._ensure_initialized()
        conn = await self._get_conn()
        cursor = await conn.execute(
            "SELECT * FROM edges WHERE user_id = ? AND relation = ? ORDER BY created_at",
            (user_id, relation),
        )
        rows = await cursor.fetchall()
        return [_row_to_edge(row) for row in rows]

    async def get_edges_to_node(self: _GraphStorageLike, user_id: str, target_node_id: str) -> list[Edge]:
        """Все рёбра входящие в указанный узел."""
        await self._ensure_initialized()
        conn = await self._get_conn()
        cursor = await conn.execute(
            "SELECT * FROM edges WHERE user_id = ? AND target_node_id = ?",
            (user_id, target_node_id),
        )
        rows = await cursor.fetchall()
        return [_row_to_edge(row) for row in rows]

    async def get_edges_from_node(self: _GraphStorageLike, user_id: str, source_node_id: str) -> list[Edge]:
        """Все рёбра исходящие из указанного узла."""
        await self._ensure_initialized()
        conn = await self._get_conn()
        cursor = await conn.execute(
            "SELECT * FROM edges WHERE user_id = ? AND source_node_id = ?",
            (user_id, source_node_id),
        )
        rows = await cursor.fetchall()
        return [_row_to_edge(row) for row in rows]


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
