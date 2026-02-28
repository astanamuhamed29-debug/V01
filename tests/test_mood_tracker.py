import asyncio

from core.graph.api import GraphAPI
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

        text = "Устал и злюсь на себя за прокрастинацию."
        replies: list[str] = []
        for _ in range(3):
            result = await processor.process(user_id="test_user", text=text, source="cli")
            replies.append(result.reply_text)

        snapshots = await storage.get_mood_snapshots("test_user", limit=3)

        assert len(snapshots) == 3
        assert "Замечаю, что" in replies[-1] or "Фиксирую нарастающее напряжение." in replies[-1]

    asyncio.run(scenario())
