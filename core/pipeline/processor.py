"""MessageProcessor — thin OODA orchestrator.

Delegates to four stage modules:
  OBSERVE → ORIENT → DECIDE → ACT

Each stage is a self-contained class with an ``async def run()`` method.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING

from config import USE_LLM
from core.context.builder import GraphContextBuilder
from core.context.session_memory import SessionMemory
from core.graph.api import GraphAPI
from core.graph.model import Edge, Node
from core.journal.storage import JournalStorage
from core.llm.embedding_service import EmbeddingService
from core.llm_client import LLMClient, MockLLMClient
from core.mood.tracker import MoodTracker
from core.parts.memory import PartsMemory
from core.pipeline.events import EventBus
from core.pipeline.stage_observe import ObserveStage
from core.pipeline.stage_orient import OrientStage
from core.pipeline.stage_decide import DecideStage
from core.pipeline.stage_act import ActStage
from core.pipeline.stage_observe import _sanitize_text  # noqa: F401 — backward compat
from core.search.qdrant_storage import QdrantVectorStorage

if TYPE_CHECKING:
    from core.analytics.calibrator import ThresholdCalibrator

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class ProcessResult:
    intent: str
    reply_text: str
    nodes: list[Node]
    edges: list[Edge]


class MessageProcessor:
    """Thin orchestrator: OBSERVE → ORIENT → DECIDE → ACT."""

    def __init__(
        self,
        graph_api: GraphAPI,
        journal: JournalStorage,
        qdrant: QdrantVectorStorage,
        session_memory: SessionMemory,
        llm_client: LLMClient | None = None,
        embedding_service: EmbeddingService | None = None,
        calibrator: "ThresholdCalibrator | None" = None,
        use_llm: bool | None = None,
        event_bus: EventBus | None = None,
    ) -> None:
        self.graph_api = graph_api
        self.journal = journal
        self.session_memory = session_memory
        self.calibrator = calibrator
        effective_llm = llm_client or MockLLMClient()
        effective_bus = event_bus or EventBus()
        effective_use_llm = USE_LLM if use_llm is None else use_llm

        self.context_builder = GraphContextBuilder(graph_api.storage, embedding_service=embedding_service)
        self.pattern_analyzer = self.context_builder.pattern_analyzer
        self.mood_tracker = MoodTracker(graph_api.storage)
        self.parts_memory = PartsMemory(graph_api.storage)

        # Expose for tests / external access
        self.llm_client = effective_llm
        self.use_llm = effective_use_llm
        self.event_bus = effective_bus
        self.embedding_service = embedding_service
        self.qdrant = qdrant
        self.live_reply_enabled: bool = os.getenv("LIVE_REPLY_ENABLED", "true").lower() == "true"

        self._calibrator_loaded: set[str] = set()

        # Build OODA stages
        self._observe = ObserveStage(
            journal=journal,
            session_memory=session_memory,
            event_bus=effective_bus,
        )
        self._orient = OrientStage(
            graph_api=graph_api,
            llm_client=effective_llm,
            embedding_service=embedding_service,
            qdrant=qdrant,
            context_builder=self.context_builder,
            use_llm=effective_use_llm,
            session_memory=session_memory,
        )
        self._decide = DecideStage(
            graph_api=graph_api,
            mood_tracker=self.mood_tracker,
            parts_memory=self.parts_memory,
        )
        self._act = ActStage(
            llm_client=effective_llm,
            session_memory=session_memory,
            event_bus=effective_bus,
        )

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    async def process_message(
        self,
        user_id: str,
        text: str,
        *,
        source: str = "cli",
        timestamp: str | None = None,
    ) -> ProcessResult:
        # Auto-load calibrator thresholds once per user
        if self.calibrator and user_id not in self._calibrator_loaded:
            try:
                await self.calibrator.load(user_id)
            except Exception as exc:
                logger.warning("ThresholdCalibrator.load failed: %s", exc)
            self._calibrator_loaded.add(user_id)

        # 1. OBSERVE — sanitise, journal, session, classify
        obs = await self._observe.run(user_id, text, source=source, timestamp=timestamp)

        # 2. ORIENT — extract, persist, embed, search, context
        ori = await self._orient.run(user_id, obs.text, obs.intent)

        # 3. DECIDE — policy, task links, conflicts, mood
        dec = await self._decide.run(
            user_id=user_id,
            created_nodes=ori.created_nodes,
            created_edges=ori.created_edges,
            retrieved_context=ori.retrieved_context,
            graph_context=ori.graph_context,
        )

        all_edges = [*ori.created_edges, *dec.created_edges]

        # 4. ACT — reply, session memory, event
        act = await self._act.run(
            user_id=user_id,
            text=obs.text,
            intent=ori.intent,
            created_nodes=ori.created_nodes,
            created_edges=all_edges,
            graph_context=ori.graph_context,
            mood_context=dec.mood_context,
            parts_context=dec.parts_context,
            retrieved_context=ori.retrieved_context,
            policy=dec.policy,
        )

        return ProcessResult(
            intent=ori.intent,
            reply_text=act.reply_text,
            nodes=ori.created_nodes,
            edges=all_edges,
        )

    # ------------------------------------------------------------------
    # Aliases & utilities kept on processor for backward compatibility
    # ------------------------------------------------------------------

    async def process(
        self,
        user_id: str,
        text: str,
        *,
        source: str = "cli",
        timestamp: str | None = None,
    ) -> ProcessResult:
        return await self.process_message(user_id=user_id, text=text, source=source, timestamp=timestamp)

    async def build_weekly_report(self, user_id: str) -> str:
        now = datetime.now(timezone.utc)
        week_ago = now - timedelta(days=7)

        snapshots = await self.graph_api.storage.get_mood_snapshots(user_id, limit=30)
        weekly = []
        for snapshot in snapshots:
            ts = snapshot.get("timestamp")
            if not ts:
                continue
            try:
                dt = datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
            except ValueError:
                continue
            if dt >= week_ago:
                weekly.append(snapshot)

        graph_context = await self.context_builder.build(user_id)
        top_parts = graph_context.get("known_parts", [])[:3]
        active_values = graph_context.get("known_values", [])[:5]

        if not weekly:
            part_line = ", ".join(p.get("name") or p.get("key") or "part" for p in top_parts) or "нет"
            value_line = ", ".join(v.get("name") or v.get("key") or "value" for v in active_values) or "нет"
            return (
                "📊 Недельный отчёт\n"
                "За последние 7 дней mood-срезов пока нет.\n"
                f"Топ частей: {part_line}\n"
                f"Активные ценности: {value_line}"
            )

        avg_valence = sum(float(s.get("valence_avg", 0.0)) for s in weekly) / len(weekly)
        avg_arousal = sum(float(s.get("arousal_avg", 0.0)) for s in weekly) / len(weekly)
        avg_dominance = sum(float(s.get("dominance_avg", 0.0)) for s in weekly) / len(weekly)

        labels: dict[str, int] = {}
        for snapshot in weekly:
            label = str(snapshot.get("dominant_label") or "").strip()
            if not label:
                continue
            labels[label] = labels.get(label, 0) + 1
        top_label = max(labels.items(), key=lambda item: item[1])[0] if labels else "не определено"

        part_line = ", ".join(
            f"{p.get('name') or p.get('key') or 'part'} ({p.get('appearances', 1)})"
            for p in top_parts
        ) or "нет"
        value_line = ", ".join(v.get("name") or v.get("key") or "value" for v in active_values) or "нет"

        return (
            "📊 Недельный отчёт\n"
            f"Срезов: {len(weekly)}\n"
            f"Среднее состояние: valence={avg_valence:.2f}, arousal={avg_arousal:.2f}, dominance={avg_dominance:.2f}\n"
            f"Чаще всего: {top_label}\n"
            f"Топ частей: {part_line}\n"
            f"Активные ценности: {value_line}"
        )
