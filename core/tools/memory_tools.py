"""Built-in tools for the SELF-OS chat model.

These tools give the reply-generating LLM access to the user's
graph, projects, insights and memory search.  Each tool is a thin
wrapper around existing storage / context APIs.
"""

from __future__ import annotations

import logging
from typing import Any, TYPE_CHECKING

from core.tools.base import Tool, ToolCallResult, ToolParameter

if TYPE_CHECKING:
    from core.context.builder import GraphContextBuilder
    from core.graph.api import GraphAPI
    from core.llm.embedding_service import EmbeddingService
    from core.search.qdrant_storage import QdrantVectorStorage, VectorSearchResult

logger = logging.getLogger(__name__)


# ─── SearchMemoryTool ─────────────────────────────────────────────

class SearchMemoryTool(Tool):
    """Semantic search across user's past messages and nodes."""

    name = "search_memory"
    description = "Поиск по памяти пользователя — прошлые записи, сообщения, ноды графа"
    parameters = [
        ToolParameter(name="query", type="string", description="Поисковый запрос"),
    ]

    def __init__(
        self,
        qdrant: "QdrantVectorStorage",
        embedding_service: "EmbeddingService | None",
        user_id: str,
    ) -> None:
        self._qdrant = qdrant
        self._embedding = embedding_service
        self._user_id = user_id

    async def execute(self, **kwargs: Any) -> ToolCallResult:
        query = kwargs.get("query", "")
        if not query:
            return ToolCallResult(tool_name=self.name, success=False, error="empty query")
        if not self._embedding:
            return ToolCallResult(tool_name=self.name, success=False, error="embeddings unavailable")

        try:
            vec = await self._embedding.embed(query)
            results = self._qdrant.search_similar(
                query_embedding=vec,
                user_id=self._user_id,
                top_k=5,
            )
            items = [
                {
                    "text": r.payload.get("text", r.payload.get("name", ""))[:200],
                    "type": r.payload.get("type", ""),
                    "score": round(r.score, 3),
                }
                for r in results
            ]
            return ToolCallResult(tool_name=self.name, success=True, data=items)
        except Exception as exc:
            return ToolCallResult(tool_name=self.name, success=False, error=str(exc))


# ─── GetProjectsTool ──────────────────────────────────────────────

class GetProjectsTool(Tool):
    """List user's active projects with their tasks."""

    name = "get_projects"
    description = "Список активных проектов пользователя и их задач"
    parameters = []

    def __init__(self, graph_api: "GraphAPI", user_id: str) -> None:
        self._api = graph_api
        self._user_id = user_id

    async def execute(self, **kwargs: Any) -> ToolCallResult:
        try:
            projects = await self._api.get_user_nodes_by_type(self._user_id, "PROJECT")
            tasks = await self._api.get_user_nodes_by_type(self._user_id, "TASK")
            edges = await self._api.storage.list_edges(self._user_id)

            # Build project → tasks map
            project_tasks: dict[str, list[str]] = {}
            for edge in edges:
                if edge.relation == "HAS_TASK":
                    project_tasks.setdefault(edge.source_node_id, []).append(
                        edge.target_node_id
                    )

            task_map = {t.id: (t.name or t.text or "")[:80] for t in tasks}

            items = []
            for p in projects:
                task_list = [
                    task_map[tid]
                    for tid in project_tasks.get(p.id, [])
                    if tid in task_map
                ]
                items.append({
                    "name": p.name or "",
                    "tasks": task_list[:10],
                })

            # Orphan tasks
            linked_task_ids = {
                tid for tids in project_tasks.values() for tid in tids
            }
            orphan = [
                task_map[t.id] for t in tasks if t.id not in linked_task_ids
            ]
            if orphan:
                items.append({"name": "(без проекта)", "tasks": orphan[:10]})

            return ToolCallResult(tool_name=self.name, success=True, data=items)
        except Exception as exc:
            return ToolCallResult(tool_name=self.name, success=False, error=str(exc))


# ─── GetInsightsTool ──────────────────────────────────────────────

class GetInsightsTool(Tool):
    """Retrieve recent system-generated insights."""

    name = "get_insights"
    description = "Получить последние инсайты системы — обнаруженные паттерны поведения и эмоций"
    parameters = [
        ToolParameter(
            name="limit", type="number",
            description="Сколько инсайтов вернуть (по умолчанию 5)",
            required=False,
        ),
    ]

    def __init__(self, graph_api: "GraphAPI", user_id: str) -> None:
        self._api = graph_api
        self._user_id = user_id

    async def execute(self, **kwargs: Any) -> ToolCallResult:
        limit = int(kwargs.get("limit", 5))
        try:
            insights = await self._api.storage.find_nodes(
                self._user_id, node_type="INSIGHT", limit=limit,
            )
            # Sort by created_at desc
            insights.sort(
                key=lambda n: n.metadata.get("created_at", n.created_at or ""),
                reverse=True,
            )
            items = [
                {
                    "title": n.name or "",
                    "description": (n.text or "")[:200],
                    "type": n.metadata.get("pattern_type", ""),
                    "severity": n.metadata.get("severity", "info"),
                    "confidence": n.metadata.get("confidence", 0),
                }
                for n in insights[:limit]
            ]
            return ToolCallResult(tool_name=self.name, success=True, data=items)
        except Exception as exc:
            return ToolCallResult(tool_name=self.name, success=False, error=str(exc))


# ─── GetMoodTrendTool ─────────────────────────────────────────────

class GetMoodTrendTool(Tool):
    """Get current mood trend and recent emotional state."""

    name = "get_mood"
    description = "Текущий эмоциональный тренд и последние состояния"
    parameters = []

    def __init__(self, graph_api: "GraphAPI", user_id: str) -> None:
        self._api = graph_api
        self._user_id = user_id

    async def execute(self, **kwargs: Any) -> ToolCallResult:
        try:
            snapshots = await self._api.storage.get_mood_snapshots(
                self._user_id, limit=5,
            )
            emotions = await self._api.storage.find_nodes(
                self._user_id, node_type="EMOTION", limit=10,
            )
            emotions.sort(
                key=lambda n: n.metadata.get("created_at", n.created_at or ""),
                reverse=True,
            )
            recent_labels = [
                n.metadata.get("label", "")
                for n in emotions[:5]
                if n.metadata.get("label")
            ]

            if snapshots:
                latest = snapshots[0]
                data = {
                    "dominant_label": latest.get("dominant_label", ""),
                    "valence_avg": round(float(latest.get("valence_avg", 0)), 2),
                    "arousal_avg": round(float(latest.get("arousal_avg", 0)), 2),
                    "recent_emotions": recent_labels,
                }
            else:
                data = {
                    "dominant_label": "",
                    "recent_emotions": recent_labels,
                }

            return ToolCallResult(tool_name=self.name, success=True, data=data)
        except Exception as exc:
            return ToolCallResult(tool_name=self.name, success=False, error=str(exc))


# ─── Factory ──────────────────────────────────────────────────────

def build_default_tools(
    graph_api: "GraphAPI",
    qdrant: "QdrantVectorStorage",
    user_id: str,
    embedding_service: "EmbeddingService | None" = None,
) -> list[Tool]:
    """Create the default tool set bound to a specific user."""
    return [
        SearchMemoryTool(qdrant=qdrant, embedding_service=embedding_service, user_id=user_id),
        GetProjectsTool(graph_api=graph_api, user_id=user_id),
        GetInsightsTool(graph_api=graph_api, user_id=user_id),
        GetMoodTrendTool(graph_api=graph_api, user_id=user_id),
    ]
