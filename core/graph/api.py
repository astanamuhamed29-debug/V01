from __future__ import annotations

import re

from core.graph.model import Edge, Node
from core.graph.storage import GraphStorage


def normalize_key(value: str) -> str:
    normalized = value.strip().lower()
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized


class GraphAPI:
    def __init__(self, storage: GraphStorage) -> None:
        self.storage = storage

    async def ensure_person_node(self, user_id: str) -> Node:
        return await self.find_or_create_node(
            user_id=user_id,
            node_type="PERSON",
            key="person:me",
            name="me",
        )

    async def create_node(
        self,
        user_id: str,
        node_type: str,
        *,
        name: str | None = None,
        text: str | None = None,
        subtype: str | None = None,
        key: str | None = None,
        metadata: dict | None = None,
    ) -> Node:
        node = Node(
            user_id=user_id,
            type=node_type,
            name=name,
            text=text,
            subtype=subtype,
            key=key,
            metadata=metadata or {},
        )
        return await self.storage.upsert_node(node)

    async def find_or_create_node(
        self,
        user_id: str,
        node_type: str,
        key: str,
        *,
        name: str | None = None,
        text: str | None = None,
        subtype: str | None = None,
        metadata: dict | None = None,
    ) -> Node:
        existing = await self.storage.find_by_key(user_id, node_type, key)
        if existing:
            return existing
        return await self.create_node(
            user_id=user_id,
            node_type=node_type,
            name=name,
            text=text,
            subtype=subtype,
            key=key,
            metadata=metadata,
        )

    async def create_edge(
        self,
        user_id: str,
        source_node_id: str,
        target_node_id: str,
        relation: str,
        *,
        metadata: dict | None = None,
    ) -> Edge:
        edge = Edge(
            user_id=user_id,
            source_node_id=source_node_id,
            target_node_id=target_node_id,
            relation=relation,
            metadata=metadata or {},
        )
        return await self.storage.add_edge(edge)

    async def apply_changes(self, user_id: str, nodes: list[Node], edges: list[Edge]) -> tuple[list[Node], list[Edge]]:
        created_nodes: list[Node] = []
        node_id_map: dict[str, str] = {}

        for node in nodes:
            if node.user_id != user_id:
                continue
            saved = await self.storage.upsert_node(node)
            node_id_map[node.id] = saved.id
            created_nodes.append(saved)

        created_edges: list[Edge] = []
        for edge in edges:
            if edge.user_id != user_id:
                continue
            source_id = node_id_map.get(edge.source_node_id, edge.source_node_id)
            target_id = node_id_map.get(edge.target_node_id, edge.target_node_id)
            saved = await self.create_edge(
                user_id=user_id,
                source_node_id=source_id,
                target_node_id=target_id,
                relation=edge.relation,
                metadata=edge.metadata,
            )
            created_edges.append(saved)

        return created_nodes, created_edges

    async def get_subgraph(self, user_id: str, node_types: list[str] | None = None) -> dict:
        nodes = await self.storage.find_nodes(user_id=user_id)
        if node_types:
            allowed = set(node_types)
            nodes = [node for node in nodes if node.type in allowed]

        node_ids = {node.id for node in nodes}
        edges = [
            edge
            for edge in await self.storage.list_edges(user_id)
            if edge.source_node_id in node_ids and edge.target_node_id in node_ids
        ]
        return {"nodes": nodes, "edges": edges}

    async def get_user_nodes_by_type(self, user_id: str, node_type: str) -> list[Node]:
        return await self.storage.find_nodes(user_id=user_id, node_type=node_type)
