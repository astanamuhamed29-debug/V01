"""Neo4jStorage — Neo4j-backed graph storage for semantic memory.

Stores long-term semantic nodes (BELIEF, NEED, VALUE, PART) and their
relationships in a Neo4j graph database.  Episodic data (NOTE, EVENT,
EMOTION, …) stays in SQLite.

This module implements the same public interface as
:class:`~core.graph.storage.GraphStorage` for the subset of operations
relevant to semantic memory, so that the consolidator pipeline can
promote nodes from SQLite → Neo4j transparently.

Configuration is via environment variables:

* ``NEO4J_URI``       – bolt://… or neo4j://… (default ``bolt://localhost:7687``)
* ``NEO4J_USER``      – (default ``neo4j``)
* ``NEO4J_PASSWORD``  – (default ``password``)
* ``NEO4J_DATABASE``  – (default ``neo4j``)

See FRONTIER_VISION_REPORT §2 and §6 — *Semantic Memory layer*.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from core.graph.model import Edge, Node, ensure_metadata_defaults

logger = logging.getLogger(__name__)

# Node types that belong in semantic (long-term) memory.
SEMANTIC_NODE_TYPES: frozenset[str] = frozenset({"BELIEF", "NEED", "VALUE", "PART"})


class Neo4jStorage:
    """Neo4j graph storage backend for long-term semantic memory.

    Parameters
    ----------
    uri:
        Bolt URI for the Neo4j instance.
    user:
        Neo4j username.
    password:
        Neo4j password.
    database:
        Neo4j database name.
    """

    def __init__(
        self,
        uri: str = "bolt://localhost:7687",
        user: str = "neo4j",
        password: str = "password",
        database: str = "neo4j",
    ) -> None:
        try:
            from neo4j import AsyncGraphDatabase  # type: ignore[import-untyped]
        except ImportError as exc:
            raise ImportError(
                "neo4j driver is required for Neo4jStorage. "
                "Install it with: pip install neo4j"
            ) from exc

        self._driver = AsyncGraphDatabase.driver(uri, auth=(user, password))
        self._database = database
        self._initialized = False

    async def close(self) -> None:
        """Close the Neo4j driver connection."""
        await self._driver.close()

    async def _ensure_initialized(self) -> None:
        """Create constraints and indexes on first use."""
        if self._initialized:
            return
        async with self._driver.session(database=self._database) as session:
            # Unique constraint on (user_id, node_type, key) for keyed nodes
            await session.run(
                "CREATE CONSTRAINT IF NOT EXISTS "
                "FOR (n:SemanticNode) REQUIRE (n.user_id, n.type, n.key) IS UNIQUE"
            )
            # Index for fast user lookups
            await session.run(
                "CREATE INDEX IF NOT EXISTS FOR (n:SemanticNode) ON (n.user_id, n.type)"
            )
            await session.run(
                "CREATE INDEX IF NOT EXISTS FOR (n:SemanticNode) ON (n.id)"
            )
        self._initialized = True

    # ── Node operations ──────────────────────────────────────────

    async def upsert_node(self, node: Node) -> Node:
        """Insert or update a semantic node in Neo4j.

        Only nodes whose ``type`` is in :data:`SEMANTIC_NODE_TYPES` should
        be stored here.  The caller is responsible for routing.
        """
        await self._ensure_initialized()
        node_metadata = ensure_metadata_defaults(dict(node.metadata))

        async with self._driver.session(database=self._database) as session:
            result = await session.run(
                """
                MERGE (n:SemanticNode {id: $id})
                ON CREATE SET
                    n.user_id    = $user_id,
                    n.type       = $type,
                    n.name       = $name,
                    n.text       = $text,
                    n.subtype    = $subtype,
                    n.key        = $key,
                    n.metadata   = $metadata_json,
                    n.created_at = $created_at,
                    n.embedding  = $embedding_json,
                    n.is_deleted = 0
                ON MATCH SET
                    n.name       = COALESCE($name, n.name),
                    n.text       = COALESCE($text, n.text),
                    n.subtype    = COALESCE($subtype, n.subtype),
                    n.metadata   = $metadata_json,
                    n.embedding  = COALESCE($embedding_json, n.embedding)
                RETURN n
                """,
                id=node.id,
                user_id=node.user_id,
                type=node.type,
                name=node.name,
                text=node.text,
                subtype=node.subtype,
                key=node.key,
                metadata_json=json.dumps(node_metadata, ensure_ascii=False),
                created_at=node.created_at,
                embedding_json=(
                    json.dumps(node.embedding) if node.embedding else None
                ),
            )
            record = await result.single()

        return _record_to_node(record["n"]) if record else node

    async def get_node(self, node_id: str) -> Node:
        """Retrieve a single node by id."""
        await self._ensure_initialized()
        async with self._driver.session(database=self._database) as session:
            result = await session.run(
                "MATCH (n:SemanticNode {id: $id}) RETURN n",
                id=node_id,
            )
            record = await result.single()
        if record is None:
            raise KeyError(f"Node not found: {node_id}")
        return _record_to_node(record["n"])

    async def find_nodes(
        self,
        user_id: str,
        node_type: str | None = None,
        name: str | None = None,
        limit: int = 500,
    ) -> list[Node]:
        """Find non-deleted nodes for a user, optionally filtered by type/name."""
        await self._ensure_initialized()

        clauses = ["n.user_id = $user_id", "(n.is_deleted IS NULL OR n.is_deleted = 0)"]
        params: dict[str, Any] = {"user_id": user_id, "limit": limit}

        if node_type:
            clauses.append("n.type = $node_type")
            params["node_type"] = node_type
        if name:
            clauses.append("n.name = $name")
            params["name"] = name

        where = " AND ".join(clauses)
        query = f"MATCH (n:SemanticNode) WHERE {where} RETURN n ORDER BY n.created_at LIMIT $limit"

        async with self._driver.session(database=self._database) as session:
            result = await session.run(query, **params)
            records = await result.data()

        return [_record_to_node(r["n"]) for r in records]

    async def find_by_key(
        self, user_id: str, node_type: str, key: str
    ) -> Node | None:
        """Find a node by its unique (user_id, type, key) triple."""
        await self._ensure_initialized()
        async with self._driver.session(database=self._database) as session:
            result = await session.run(
                """
                MATCH (n:SemanticNode {user_id: $user_id, type: $type, key: $key})
                RETURN n
                """,
                user_id=user_id,
                type=node_type,
                key=key,
            )
            record = await result.single()
        return _record_to_node(record["n"]) if record else None

    async def soft_delete_node(self, node_id: str) -> None:
        """Mark a node as deleted without physically removing it."""
        await self._ensure_initialized()
        async with self._driver.session(database=self._database) as session:
            await session.run(
                "MATCH (n:SemanticNode {id: $id}) SET n.is_deleted = 1",
                id=node_id,
            )

    # ── Edge (relationship) operations ───────────────────────────

    async def add_edge(self, edge: Edge) -> Edge:
        """Create or return an existing relationship between two semantic nodes."""
        await self._ensure_initialized()
        async with self._driver.session(database=self._database) as session:
            result = await session.run(
                """
                MATCH (a:SemanticNode {id: $source_id})
                MATCH (b:SemanticNode {id: $target_id})
                MERGE (a)-[r:RELATES {
                    user_id: $user_id,
                    relation: $relation
                }]->(b)
                ON CREATE SET
                    r.id          = $id,
                    r.metadata    = $metadata_json,
                    r.created_at  = $created_at
                RETURN r
                """,
                source_id=edge.source_node_id,
                target_id=edge.target_node_id,
                user_id=edge.user_id,
                relation=edge.relation,
                id=edge.id,
                metadata_json=json.dumps(edge.metadata, ensure_ascii=False),
                created_at=edge.created_at,
            )
            record = await result.single()
        if record is None:
            logger.warning("add_edge: source or target node not found")
            return edge
        return _record_to_edge(record["r"], edge.source_node_id, edge.target_node_id)

    async def list_edges(self, user_id: str) -> list[Edge]:
        """Return all relationships for a user."""
        await self._ensure_initialized()
        async with self._driver.session(database=self._database) as session:
            result = await session.run(
                """
                MATCH (a:SemanticNode)-[r:RELATES {user_id: $user_id}]->(b:SemanticNode)
                RETURN r, a.id AS source_id, b.id AS target_id
                ORDER BY r.created_at
                """,
                user_id=user_id,
            )
            records = await result.data()
        return [
            _record_to_edge(r["r"], r["source_id"], r["target_id"])
            for r in records
        ]

    async def get_edges_from_node(
        self, user_id: str, source_node_id: str
    ) -> list[Edge]:
        """Return all outgoing relationships from a node."""
        await self._ensure_initialized()
        async with self._driver.session(database=self._database) as session:
            result = await session.run(
                """
                MATCH (a:SemanticNode {id: $source_id})-[r:RELATES {user_id: $user_id}]->(b:SemanticNode)
                RETURN r, a.id AS source_id, b.id AS target_id
                """,
                source_id=source_node_id,
                user_id=user_id,
            )
            records = await result.data()
        return [
            _record_to_edge(r["r"], r["source_id"], r["target_id"])
            for r in records
        ]

    async def get_edges_to_node(
        self, user_id: str, target_node_id: str
    ) -> list[Edge]:
        """Return all incoming relationships to a node."""
        await self._ensure_initialized()
        async with self._driver.session(database=self._database) as session:
            result = await session.run(
                """
                MATCH (a:SemanticNode)-[r:RELATES {user_id: $user_id}]->(b:SemanticNode {id: $target_id})
                RETURN r, a.id AS source_id, b.id AS target_id
                """,
                target_id=target_node_id,
                user_id=user_id,
            )
            records = await result.data()
        return [
            _record_to_edge(r["r"], r["source_id"], r["target_id"])
            for r in records
        ]

    # ── Merge / consolidation ────────────────────────────────────

    async def merge_nodes(
        self,
        user_id: str,
        source_node_ids: list[str],
        target_node: Node,
    ) -> Node:
        """Merge several nodes into *target_node* within Neo4j.

        Re-points all relationships, removes self-loops, soft-deletes sources.
        """
        if not source_node_ids:
            return await self.upsert_node(target_node)

        await self._ensure_initialized()
        saved = await self.upsert_node(target_node)

        async with self._driver.session(database=self._database) as session:
            for source_id in source_node_ids:
                # Re-point outgoing edges
                await session.run(
                    """
                    MATCH (s:SemanticNode {id: $source_id})-[r:RELATES]->(b:SemanticNode)
                    WHERE b.id <> $target_id
                    MATCH (t:SemanticNode {id: $target_id})
                    CREATE (t)-[r2:RELATES]->(b)
                    SET r2 = properties(r)
                    DELETE r
                    """,
                    source_id=source_id,
                    target_id=saved.id,
                )
                # Re-point incoming edges
                await session.run(
                    """
                    MATCH (a:SemanticNode)-[r:RELATES]->(s:SemanticNode {id: $source_id})
                    WHERE a.id <> $target_id
                    MATCH (t:SemanticNode {id: $target_id})
                    CREATE (a)-[r2:RELATES]->(t)
                    SET r2 = properties(r)
                    DELETE r
                    """,
                    source_id=source_id,
                    target_id=saved.id,
                )
                # Remove any remaining self-loops / edges from source
                await session.run(
                    """
                    MATCH (s:SemanticNode {id: $source_id})-[r:RELATES]-(any)
                    DELETE r
                    """,
                    source_id=source_id,
                )
                # Soft-delete source
                await session.run(
                    "MATCH (n:SemanticNode {id: $id}) SET n.is_deleted = 1",
                    id=source_id,
                )

            # Remove self-loops on target
            await session.run(
                """
                MATCH (t:SemanticNode {id: $id})-[r:RELATES]->(t)
                DELETE r
                """,
                id=saved.id,
            )

        return saved

    # ── Graph traversal (Neo4j advantage) ────────────────────────

    async def find_paths(
        self,
        user_id: str,
        start_node_id: str,
        end_node_id: str,
        max_depth: int = 5,
    ) -> list[list[str]]:
        """Find all simple paths between two nodes (up to *max_depth* hops).

        Returns a list of node-id paths.  This is O(log N) in Neo4j vs O(N)
        in SQLite — the primary reason for the migration.
        """
        await self._ensure_initialized()
        async with self._driver.session(database=self._database) as session:
            result = await session.run(
                """
                MATCH path = shortestPath(
                    (a:SemanticNode {id: $start_id})-[:RELATES*1.."""
                + str(max_depth)
                + """]->(b:SemanticNode {id: $end_id})
                )
                WHERE ALL(n IN nodes(path) WHERE n.user_id = $user_id)
                RETURN [n IN nodes(path) | n.id] AS node_ids
                """,
                start_id=start_node_id,
                end_id=end_node_id,
                user_id=user_id,
            )
            records = await result.data()
        return [r["node_ids"] for r in records]

    async def get_neighborhood(
        self,
        user_id: str,
        node_id: str,
        depth: int = 2,
    ) -> list[Node]:
        """Return all nodes within *depth* hops of *node_id*.

        .. note::
           Requires the APOC plugin to be installed in the Neo4j instance.
           See https://neo4j.com/labs/apoc/
        """
        await self._ensure_initialized()
        async with self._driver.session(database=self._database) as session:
            result = await session.run(
                """
                MATCH (start:SemanticNode {id: $node_id, user_id: $user_id})
                CALL apoc.path.subgraphNodes(start, {maxLevel: $depth})
                YIELD node
                WHERE node.user_id = $user_id
                  AND (node.is_deleted IS NULL OR node.is_deleted = 0)
                RETURN node
                """,
                node_id=node_id,
                user_id=user_id,
                depth=depth,
            )
            records = await result.data()
        return [_record_to_node(r["node"]) for r in records]

    async def count_nodes(self, user_id: str) -> int:
        """Total number of semantic nodes for a user."""
        await self._ensure_initialized()
        async with self._driver.session(database=self._database) as session:
            result = await session.run(
                """
                MATCH (n:SemanticNode {user_id: $user_id})
                WHERE n.is_deleted IS NULL OR n.is_deleted = 0
                RETURN count(n) AS cnt
                """,
                user_id=user_id,
            )
            record = await result.single()
        return record["cnt"] if record else 0

    async def delete_all_user_data(self, user_id: str) -> int:
        """Physically delete all nodes and relationships for a user.

        Intended for testing and GDPR data-deletion requests.
        Returns the number of nodes removed.
        """
        await self._ensure_initialized()
        async with self._driver.session(database=self._database) as session:
            result = await session.run(
                """
                MATCH (n:SemanticNode {user_id: $user_id})
                DETACH DELETE n
                RETURN count(n) AS cnt
                """,
                user_id=user_id,
            )
            record = await result.single()
        return record["cnt"] if record else 0


# ── Helpers ──────────────────────────────────────────────────────


def _record_to_node(props: Any) -> Node:
    """Convert a Neo4j node record to a :class:`Node`."""
    metadata = json.loads(props.get("metadata", "{}"))
    embedding_raw = props.get("embedding")
    embedding = json.loads(embedding_raw) if embedding_raw else None
    return Node(
        id=props["id"],
        user_id=props["user_id"],
        type=props["type"],
        name=props.get("name"),
        text=props.get("text"),
        subtype=props.get("subtype"),
        key=props.get("key"),
        metadata=metadata,
        created_at=props.get("created_at", ""),
        embedding=embedding,
    )


def _record_to_edge(
    props: Any,
    source_id: str,
    target_id: str,
) -> Edge:
    """Convert a Neo4j relationship record to an :class:`Edge`."""
    metadata = json.loads(props.get("metadata", "{}"))
    return Edge(
        id=props.get("id", ""),
        user_id=props.get("user_id", ""),
        source_node_id=source_id,
        target_node_id=target_id,
        relation=props.get("relation", "RELATES_TO"),
        metadata=metadata,
        created_at=props.get("created_at", ""),
    )
