import asyncio

from core.context.session_memory import SessionMemory
from core.graph.api import GraphAPI
from core.graph.storage import GraphStorage
from core.journal.storage import JournalStorage
from core.pipeline.processor import MessageProcessor


class _NoopQdrant:
    def upsert_embeddings_batch(self, points):
        return

    def search_similar(self, *args, **kwargs):
        return []


class LiveReplyFallbackLLMClient:
    def __init__(self) -> None:
        self.live_reply_calls = 0

    async def classify_intent(self, text: str) -> str:
        return "META"

    async def extract_all(self, text: str, intent: str, graph_hints: dict | None = None):
        return {"nodes": [], "edges": []}

    async def extract_semantic(self, text: str, intent: str):
        return {"nodes": [], "edges": []}

    async def extract_parts(self, text: str, intent: str):
        return {"nodes": [], "edges": []}

    async def extract_emotion(self, text: str, intent: str):
        return {"nodes": [], "edges": []}

    async def generate_live_reply(
        self,
        user_text: str,
        intent: str,
        mood_context: dict | None,
        parts_context: list[dict] | None,
        graph_context: dict | None,
    ) -> str:
        self.live_reply_calls += 1
        return ""


class LiveReplyDisabledLLMClient(LiveReplyFallbackLLMClient):
    pass


def test_live_reply_fallback(tmp_path, monkeypatch):
    async def scenario() -> None:
        monkeypatch.setenv("LIVE_REPLY_ENABLED", "true")

        db_path = tmp_path / "live_reply_fallback.db"
        api = GraphAPI(GraphStorage(db_path=db_path))
        journal = JournalStorage(db_path=db_path)
        llm_client = LiveReplyFallbackLLMClient()
        processor = MessageProcessor(
            graph_api=api,
            journal=journal,
            qdrant=_NoopQdrant(),
            session_memory=SessionMemory(),
            llm_client=llm_client,
        )

        try:
            result = await processor.process_message(user_id="me", text="в чем твоя польза", source="cli")

            assert llm_client.live_reply_calls == 1
            assert result.reply_text.startswith("Слышу")
        finally:
            await api.storage.close()

    asyncio.run(scenario())


def test_live_reply_disabled(tmp_path, monkeypatch):
    async def scenario() -> None:
        monkeypatch.setenv("LIVE_REPLY_ENABLED", "false")

        db_path = tmp_path / "live_reply_disabled.db"
        api = GraphAPI(GraphStorage(db_path=db_path))
        journal = JournalStorage(db_path=db_path)
        llm_client = LiveReplyDisabledLLMClient()
        processor = MessageProcessor(
            graph_api=api,
            journal=journal,
            qdrant=_NoopQdrant(),
            session_memory=SessionMemory(),
            llm_client=llm_client,
        )

        try:
            result = await processor.process_message(user_id="me", text="в чем твоя польза", source="cli")

            assert llm_client.live_reply_calls == 0
            assert result.reply_text.startswith("Слышу")
        finally:
            await api.storage.close()

    asyncio.run(scenario())