from __future__ import annotations

import logging
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
    ) -> Edge | None:
        try:
            edge = Edge(
                user_id=user_id,
                source_node_id=source_node_id,
                target_node_id=target_node_id,
                relation=relation,
                metadata=metadata or {},
            )
            return await self.storage.add_edge(edge)
        except Exception as exc:
            logging.getLogger(__name__).warning(
                "create_edge failed %s→%s [%s]: %s",
                source_node_id,
                target_node_id,
                relation,
                exc,
            )
            return None

    async def apply_changes(self, user_id: str, nodes: list[Node], edges: list[Edge]) -> tuple[list[Node], list[Edge]]:
        node_id_map: dict[str, str] = {}
        nodes_data: list[tuple[str, Node, dict]] = []

        for node in nodes:
            if node.user_id != user_id:
                continue

            original_id = node.id

            if node.key:
                node = Node(
                    id=node.id,
                    user_id=node.user_id,
                    type=node.type,
                    name=node.name,
                    text=node.text,
                    subtype=node.subtype,
                    key=normalize_key(node.key),
                    metadata=node.metadata,
                    created_at=node.created_at,
                )

            if node.key:
                existing = await self.storage.find_by_key(user_id, node.type, node.key)
                if existing:
                    node = Node(
                        id=existing.id,
                        user_id=user_id,
                        type=node.type,
                        name=node.name or existing.name,
                        text=node.text or existing.text,
                        subtype=node.subtype or existing.subtype,
                        key=node.key,
                        metadata={**existing.metadata, **node.metadata},
                        created_at=existing.created_at,
                    )

            node_metadata = dict(node.metadata)
            if node.type == "EMOTION" and "created_at" not in node_metadata:
                from datetime import datetime, timezone

                node_metadata["created_at"] = datetime.now(timezone.utc).isoformat()

            nodes_data.append((original_id, node, node_metadata))

        saved_nodes = (
            await self.storage.upsert_nodes_batch([(node, metadata) for _, node, metadata in nodes_data])
            if nodes_data
            else []
        )

        created_nodes_by_id: dict[str, Node] = {}
        for (original_id, _, _), saved in zip(nodes_data, saved_nodes):
            node_id_map[original_id] = saved.id
            node_id_map[saved.id] = saved.id
            created_nodes_by_id[saved.id] = saved

        created_nodes = list(created_nodes_by_id.values())

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
            if saved:
                created_edges.append(saved)

        return created_nodes, created_edges

    async def get_user_nodes_by_type(self, user_id: str, node_type: str) -> list[Node]:
        return await self.storage.find_nodes(user_id=user_id, node_type=node_type)
