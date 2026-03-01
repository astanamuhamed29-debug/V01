"""OODA — DECIDE stage.

Selects policy, links tasks to projects, detects part–value conflicts,
updates mood tracker.
"""

from __future__ import annotations

import logging

from core.graph.api import GraphAPI
from core.graph.model import Edge, Node
from core.mood.tracker import MoodTracker
from core.parts.memory import PartsMemory
from core.search.qdrant_storage import VectorSearchResult

logger = logging.getLogger(__name__)


class DecideResult:
    __slots__ = (
        "policy", "mood_context", "parts_context",
        "created_edges",
    )

    def __init__(
        self,
        policy: str,
        mood_context: dict,
        parts_context: list[dict],
        created_edges: list[Edge],
    ) -> None:
        self.policy = policy
        self.mood_context = mood_context
        self.parts_context = parts_context
        self.created_edges = created_edges


class DecideStage:
    """Policy selection, task→project linking, part–value conflict detection, mood update."""

    def __init__(
        self,
        graph_api: GraphAPI,
        mood_tracker: MoodTracker,
        parts_memory: PartsMemory,
    ) -> None:
        self.graph_api = graph_api
        self.mood_tracker = mood_tracker
        self.parts_memory = parts_memory

    async def run(
        self,
        user_id: str,
        created_nodes: list[Node],
        created_edges: list[Edge],
        retrieved_context: list[VectorSearchResult],
        graph_context: dict,
    ) -> DecideResult:
        extra_edges: list[Edge] = []

        # --- Policy selection ---
        has_part = any(n.type == "PART" for n in created_nodes)
        has_value = any(n.type == "VALUE" for n in created_nodes)
        low_valence = any(
            float(n.metadata.get("pad_v", n.metadata.get("valence", 0))) < -0.5
            for n in created_nodes
            if n.type == "EMOTION"
        )
        top_score = retrieved_context[0].score if retrieved_context else 0.0

        if has_part and has_value:
            policy = "IFS_RESOLVE"
        elif low_valence:
            policy = "VALIDATE"
        elif top_score > 0.85:
            policy = "PATTERN_INSIGHT"
        else:
            policy = "REFLECT"

        # --- Task → Project linking ---
        tasks = [n for n in created_nodes if n.type == "TASK"]
        current_projects = [n for n in created_nodes if n.type == "PROJECT"]
        if tasks and current_projects:
            for task in tasks:
                edge = await self.graph_api.create_edge(
                    user_id=user_id,
                    source_node_id=current_projects[0].id,
                    target_node_id=task.id,
                    relation="HAS_TASK",
                )
                if edge:
                    extra_edges.append(edge)
        elif tasks and not current_projects:
            all_projects = await self.graph_api.get_user_nodes_by_type(user_id, "PROJECT")
            all_projects_sorted = sorted(all_projects, key=lambda n: n.created_at or "", reverse=True)
            if all_projects_sorted:
                for task in tasks:
                    edge = await self.graph_api.create_edge(
                        user_id=user_id,
                        source_node_id=all_projects_sorted[0].id,
                        target_node_id=task.id,
                        relation="HAS_TASK",
                    )
                    if edge:
                        extra_edges.append(edge)

        # --- Parts memory + conflict detection ---
        part_nodes = [n for n in created_nodes if n.type == "PART"]
        value_nodes = [n for n in created_nodes if n.type == "VALUE"]
        parts_context: list[dict] = []
        for part in part_nodes:
            history = await self.parts_memory.register_appearance(user_id, part)
            if history.get("part"):
                parts_context.append(history)

        if part_nodes and value_nodes:
            for part in part_nodes:
                for value in value_nodes:
                    conflict_edge = await self.graph_api.create_edge(
                        user_id=user_id,
                        source_node_id=part.id,
                        target_node_id=value.id,
                        relation="CONFLICTS_WITH",
                        metadata={"auto": "session_part_value_conflict"},
                    )
                    if conflict_edge:
                        extra_edges.append(conflict_edge)
                        graph_context["session_conflict"] = True

        # --- Mood update ---
        emotion_nodes = [n for n in created_nodes if n.type == "EMOTION"]
        mood_context = await self.mood_tracker.update(user_id, emotion_nodes)

        return DecideResult(
            policy=policy,
            mood_context=mood_context,
            parts_context=parts_context,
            created_edges=extra_edges,
        )
