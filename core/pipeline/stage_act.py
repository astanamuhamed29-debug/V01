"""OODA — ACT stage.

Generates reply (minimal + LLM live), updates session memory, publishes event.
"""

from __future__ import annotations

import logging
import os

from core.pipeline.reply_minimal import generate_reply
from core.context.session_memory import SessionMemory
from core.graph.model import Edge, Node
from core.llm_client import LLMClient
from core.pipeline.events import EventBus
from core.search.qdrant_storage import VectorSearchResult

logger = logging.getLogger(__name__)


class ActResult:
    __slots__ = ("reply_text",)

    def __init__(self, reply_text: str) -> None:
        self.reply_text = reply_text


class ActStage:
    """Reply generation → session memory update → event publish."""

    def __init__(
        self,
        llm_client: LLMClient,
        session_memory: SessionMemory,
        event_bus: EventBus,
    ) -> None:
        self.llm_client = llm_client
        self.session_memory = session_memory
        self.event_bus = event_bus
        self.live_reply_enabled: bool = os.getenv("LIVE_REPLY_ENABLED", "true").lower() == "true"

    async def run(
        self,
        user_id: str,
        text: str,
        intent: str,
        created_nodes: list[Node],
        created_edges: list[Edge],
        graph_context: dict,
        mood_context: dict,
        parts_context: list[dict],
        retrieved_context: list[VectorSearchResult],
        policy: str,
    ) -> ActResult:
        session_ctx = self.session_memory.get_context(user_id)

        effective_intent = intent if intent != "UNKNOWN" else "REFLECTION"
        reply_text = generate_reply(
            text=text,
            intent=effective_intent,
            extracted_structures={
                "nodes": [*created_nodes],
                "edges": [*created_edges],
            },
            mood_context=mood_context,
            parts_context=parts_context,
            graph_context=graph_context,
            retrieved_context=retrieved_context,
            session_context=session_ctx,
            policy=policy,
        )

        live_reply = await self._generate_live_reply_safe(
            text=text,
            intent=intent,
            graph_context=graph_context,
            mood_context=mood_context,
            parts_context=parts_context,
            retrieved_context=retrieved_context,
            session_context=session_ctx,
            policy=policy,
        )
        final_reply = live_reply if live_reply and live_reply.strip() else reply_text

        self.session_memory.add_message(user_id, text, role="user")
        self.session_memory.add_message(user_id, final_reply, role="assistant")

        self.event_bus.publish(
            "pipeline.processed",
            {
                "user_id": user_id,
                "intent": intent,
                "nodes": len(created_nodes),
                "edges": len(created_edges),
            },
        )

        return ActResult(reply_text=final_reply)

    # ------------------------------------------------------------------

    async def _generate_live_reply_safe(
        self,
        text: str,
        intent: str,
        graph_context: dict,
        mood_context: dict | None = None,
        parts_context: list[dict] | None = None,
        retrieved_context: list[VectorSearchResult] | None = None,
        session_context: list[dict] | None = None,
        policy: str = "REFLECT",
    ) -> str:
        if not self.live_reply_enabled:
            return ""
        try:
            runtime_context = dict(graph_context)
            runtime_context["retrieved_context"] = [r.payload for r in (retrieved_context or [])[:3]]
            runtime_context["session_context"] = (session_context or [])[-5:]
            runtime_context["policy"] = policy
            return await self.llm_client.generate_live_reply(
                user_text=text,
                intent=intent,
                mood_context=mood_context,
                parts_context=parts_context,
                graph_context=runtime_context,
            )
        except Exception as exc:
            logger.warning("live_reply_safe failed: %s", exc)
            return ""
