from __future__ import annotations

import logging
from datetime import datetime, timezone

from core.graph.model import Node
from core.graph.storage import GraphStorage

logger = logging.getLogger(__name__)

IFS_NAMES_RU = {
    "critic": "Критик",
    "protector": "Защитник",
    "exile": "Изгнанник",
    "manager": "Менеджер",
    "firefighter": "Пожарный",
    "inner_child": "Внутренний ребёнок",
}


class PartsMemory:
    def __init__(self, storage: GraphStorage) -> None:
        self.storage = storage

    async def get_known_parts(self, user_id: str) -> list[Node]:
        """Все PART-узлы пользователя из графа."""
        return await self.storage.find_nodes(user_id=user_id, node_type="PART")

    async def get_part_history(self, user_id: str, part_key: str) -> dict:
        """
        Возвращает:
        - part: Node или None
        - appearances: количество раз когда эта часть появлялась
        - last_seen: дата последнего появления
        - first_seen: дата первого появления
        """
        part = await self.storage.find_by_key(user_id, "PART", part_key)
        if not part:
            return {"part": None, "appearances": 0, "last_seen": None, "first_seen": None}

        appearances = int(part.metadata.get("appearances", 1))
        last_seen = part.metadata.get("last_seen") or part.created_at
        first_seen = part.metadata.get("first_seen") or part.created_at

        return {
            "part": part,
            "appearances": appearances,
            "last_seen": last_seen,
            "first_seen": first_seen,
        }

    async def register_appearance(self, user_id: str, part_node: Node) -> Node:
        """
        Вызывается при каждом появлении части.
        Обновляет счётчик appearances и last_seen в metadata.
        Дедупликация по key уже работает в apply_changes.
        """
        if not part_node.key:
            return part_node

        existing = await self.storage.find_by_key(user_id, "PART", part_node.key)
        now = datetime.now(timezone.utc).isoformat()

        if existing:
            appearances = int(existing.metadata.get("appearances", 0)) + 1
            updated_metadata = {
                **existing.metadata,
                "appearances": appearances,
                "last_seen": now,
                "first_seen": existing.metadata.get("first_seen") or existing.created_at,
                "voice": part_node.metadata.get("voice") or existing.metadata.get("voice", ""),
            }
            updated = Node(
                id=existing.id,
                user_id=user_id,
                type="PART",
                subtype=existing.subtype,
                name=existing.name,
                text=part_node.text or existing.text,
                key=existing.key,
                metadata=updated_metadata,
                created_at=existing.created_at,
            )
            return await self.storage.upsert_node(updated)

        part_metadata = {
            **part_node.metadata,
            "appearances": 1,
            "first_seen": now,
            "last_seen": now,
        }
        first = Node(
            id=part_node.id,
            user_id=part_node.user_id,
            type="PART",
            subtype=part_node.subtype,
            name=part_node.name,
            text=part_node.text,
            key=part_node.key,
            metadata=part_metadata,
            created_at=part_node.created_at,
        )
        return await self.storage.upsert_node(first)
