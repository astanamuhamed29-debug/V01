import asyncio
import json
from datetime import datetime, timedelta, timezone

from core.context.session_memory import SessionMemory
from core.graph.api import GraphAPI
from core.graph.model import Node
from core.graph.storage import GraphStorage
from core.journal.storage import JournalStorage
from core.pipeline.processor import MessageProcessor


class _NoopQdrant:
    def upsert_embeddings_batch(self, points):
        return

    def search_similar(self, *args, **kwargs):
        return []


class _EmotionLLMClient:
    """LLM-мок: возвращает эмоции «усталость» и «злость»."""

    async def classify_intent(self, text):
        return "FEELING_REPORT"

    async def extract_all(self, text, intent, graph_hints=None):
        return json.dumps({
            "_reasoning": {
                "situation": "Прокрастинация",
                "appraisal": "Самообвинение",
                "affect": "Усталость и злость",
                "defenses": "Критик",
                "core_needs": "Отдых"
            },
            "intent": "FEELING_REPORT",
            "nodes": [
                {"id": "n1", "type": "EMOTION",
                 "metadata": {"label": "усталость", "valence": -0.5, "arousal": -0.3,
                              "dominance": -0.4, "intensity": 0.7}},
                {"id": "n2", "type": "EMOTION",
                 "metadata": {"label": "злость", "valence": -0.7, "arousal": 0.6,
                              "dominance": -0.3, "intensity": 0.8}},
            ],
            "edges": [
                {"source_node_id": "person:me", "target_node_id": "n1", "relation": "FEELS"},
                {"source_node_id": "person:me", "target_node_id": "n2", "relation": "FEELS"},
            ]
        }, ensure_ascii=False)

    async def extract_semantic(self, text, intent):
        return {"nodes": [], "edges": []}

    async def extract_parts(self, text, intent):
        return {"nodes": [], "edges": []}

    async def extract_emotion(self, text, intent):
        return {"nodes": [], "edges": []}

    async def arbitrate_emotion(self, text, system_prompt):
        return {"emotions": []}

    async def generate_live_reply(self, user_text, intent, mood_context, parts_context, graph_context):
        return ""


def test_mood_snapshots_and_trend_reply(tmp_path):
    async def scenario() -> None:
        db_path = tmp_path / "test.db"
        storage = GraphStorage(db_path=db_path)
        api = GraphAPI(storage)
        journal = JournalStorage(db_path=db_path)
        processor = MessageProcessor(
            graph_api=api,
            journal=journal,
            qdrant=_NoopQdrant(),
            session_memory=SessionMemory(),
            llm_client=_EmotionLLMClient(),
        )

        try:
            text = "Устал и злюсь на себя за прокрастинацию."
            replies: list[str] = []
            for _ in range(3):
                result = await processor.process(user_id="test_user", text=text, source="cli")
                replies.append(result.reply_text)

            snapshots = await storage.get_mood_snapshots("test_user", limit=3)

            assert len(snapshots) == 1
            assert "Замечаю, что" in replies[-1] or "Фиксирую нарастающее напряжение." in replies[-1]
        finally:
            await storage.close()

    asyncio.run(scenario())


def test_mood_tracker_temporal_weights_favor_recent(tmp_path):
    async def scenario() -> None:
        db_path = tmp_path / "weighted.db"
        storage = GraphStorage(db_path=db_path)

        now = datetime.now(timezone.utc)
        old_ts = (now - timedelta(days=90)).isoformat()
        new_ts = (now - timedelta(days=1)).isoformat()

        try:
            await storage.upsert_node(
                Node(
                    user_id="u_weight",
                    type="EMOTION",
                    key="emotion:стыд:old",
                    metadata={
                        "label": "стыд",
                        "valence": -0.9,
                        "arousal": 0.5,
                        "dominance": -0.7,
                        "intensity": 0.8,
                        "created_at": old_ts,
                    },
                    created_at=old_ts,
                )
            )
            await storage.upsert_node(
                Node(
                    user_id="u_weight",
                    type="EMOTION",
                    key="emotion:спокойствие:new",
                    metadata={
                        "label": "спокойствие",
                        "valence": 0.4,
                        "arousal": -0.2,
                        "dominance": 0.4,
                        "intensity": 0.4,
                        "created_at": new_ts,
                    },
                    created_at=new_ts,
                )
            )

            tracker = MessageProcessor(
                graph_api=GraphAPI(storage),
                journal=JournalStorage(db_path=db_path),
                qdrant=_NoopQdrant(),
                session_memory=SessionMemory(),
            ).mood_tracker

            snapshot = await tracker.update("u_weight", [])
            assert snapshot is not None
            assert snapshot["valence_avg"] > 0
        finally:
            await storage.close()

    asyncio.run(scenario())
