from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from core.graph.model import Edge, Node


class GraphStorage:
    def __init__(self, db_path: str | Path = "data/self_os.db") -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.executescript(
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
                """
            )

    def upsert_node(self, node: Node) -> Node:
        with self._connect() as conn:
            if node.key:
                existing = conn.execute(
                    """
                    SELECT * FROM nodes
                    WHERE user_id = ? AND type = ? AND key = ?
                    """,
                    (node.user_id, node.type, node.key),
                ).fetchone()
                if existing:
                    conn.execute(
                        """
                        UPDATE nodes
                        SET name = COALESCE(?, name),
                            text = COALESCE(?, text),
                            subtype = COALESCE(?, subtype),
                            metadata_json = ?
                        WHERE id = ?
                        """,
                        (
                            node.name,
                            node.text,
                            node.subtype,
                            json.dumps(node.metadata, ensure_ascii=False),
                            existing["id"],
                        ),
                    )
                    return self.get_node(existing["id"])

            conn.execute(
                """
                INSERT INTO nodes (id, user_id, type, name, text, subtype, key, metadata_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    node.id,
                    node.user_id,
                    node.type,
                    node.name,
                    node.text,
                    node.subtype,
                    node.key,
                    json.dumps(node.metadata, ensure_ascii=False),
                    node.created_at,
                ),
            )
            return node

    def add_edge(self, edge: Edge) -> Edge:
        with self._connect() as conn:
            existing = conn.execute(
                """
                SELECT id FROM edges
                WHERE user_id = ? AND source_node_id = ? AND target_node_id = ? AND relation = ?
                """,
                (edge.user_id, edge.source_node_id, edge.target_node_id, edge.relation),
            ).fetchone()
            if existing:
                return self.get_edge(existing["id"])

            conn.execute(
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
            return edge

    def get_node(self, node_id: str) -> Node:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM nodes WHERE id = ?", (node_id,)).fetchone()
        if row is None:
            raise KeyError(f"Node not found: {node_id}")
        return _row_to_node(row)

    def get_edge(self, edge_id: str) -> Edge:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM edges WHERE id = ?", (edge_id,)).fetchone()
        if row is None:
            raise KeyError(f"Edge not found: {edge_id}")
        return _row_to_edge(row)

    def find_nodes(
        self,
        user_id: str,
        node_type: str | None = None,
        name: str | None = None,
    ) -> list[Node]:
        query = "SELECT * FROM nodes WHERE user_id = ?"
        params: list[str] = [user_id]
        if node_type:
            query += " AND type = ?"
            params.append(node_type)
        if name:
            query += " AND name = ?"
            params.append(name)
        query += " ORDER BY created_at"

        with self._connect() as conn:
            rows = conn.execute(query, params).fetchall()
        return [_row_to_node(row) for row in rows]

    def find_by_key(self, user_id: str, node_type: str, key: str) -> Node | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM nodes WHERE user_id = ? AND type = ? AND key = ?",
                (user_id, node_type, key),
            ).fetchone()
        return _row_to_node(row) if row else None

    def list_edges(self, user_id: str) -> list[Edge]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM edges WHERE user_id = ? ORDER BY created_at",
                (user_id,),
            ).fetchall()
        return [_row_to_edge(row) for row in rows]


def _row_to_node(row: sqlite3.Row) -> Node:
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


def _row_to_edge(row: sqlite3.Row) -> Edge:
    return Edge(
        id=row["id"],
        user_id=row["user_id"],
        source_node_id=row["source_node_id"],
        target_node_id=row["target_node_id"],
        relation=row["relation"],
        metadata=json.loads(row["metadata_json"]),
        created_at=row["created_at"],
    )
