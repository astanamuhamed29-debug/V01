import asyncio
from datetime import datetime, timedelta, timezone

from core.graph.api import GraphAPI
from core.graph.model import Node
from core.graph.storage import GraphStorage
from core.journal.storage import JournalStorage
from core.pipeline.processor import MessageProcessor


def test_mood_snapshots_and_trend_reply(tmp_path):
    async def scenario() -> None:
        db_path = tmp_path / "test.db"
        storage = GraphStorage(db_path=db_path)
        api = GraphAPI(storage)
        journal = JournalStorage(db_path=db_path)
        processor = MessageProcessor(graph_api=api, journal=journal, use_llm=False)

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
                use_llm=False,
            ).mood_tracker

            snapshot = await tracker.update("u_weight", [])
            assert snapshot is not None
            assert snapshot["valence_avg"] > 0
        finally:
            await storage.close()

    asyncio.run(scenario())
