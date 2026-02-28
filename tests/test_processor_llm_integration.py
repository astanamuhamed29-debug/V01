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


class FencedLLMClient:
    def __init__(self) -> None:
        self.extract_all_calls = 0

    async def classify_intent(self, text: str) -> str:
        return "FEELING_REPORT"

    async def extract_all(self, text: str, intent: str, graph_hints: dict | None = None):
        self.extract_all_calls += 1
        return """```json
{"intent":"FEELING_REPORT","nodes":[{"id":"n1","type":"PROJECT","name":"SELF-OS","key":"project:self-os"},{"id":"n2","type":"BELIEF","text":"я не вывезу проект","key":"belief:я не вывезу проект"},{"id":"n3","type":"EMOTION","metadata":{"valence":-0.7,"arousal":0.8,"label":"fear"}},{"id":"n4","type":"SOMA","metadata":{"location":"грудь","sensation":"тяжесть"}}],"edges":[{"source_node_id":"person:me","target_node_id":"n1","relation":"OWNS_PROJECT"},{"source_node_id":"person:me","target_node_id":"n2","relation":"HOLDS_BELIEF"},{"source_node_id":"person:me","target_node_id":"n3","relation":"FEELS"},{"source_node_id":"n3","target_node_id":"n1","relation":"EMOTION_ABOUT"},{"source_node_id":"n3","target_node_id":"n4","relation":"EXPRESSED_AS"}]}
```"""

    async def extract_semantic(self, text: str, intent: str):
        return {"nodes": [], "edges": []}

    async def extract_parts(self, text: str, intent: str):
        return {"nodes": [], "edges": []}

    async def extract_emotion(self, text: str, intent: str):
        return {
            "nodes": [
                {
                    "id": "e1",
                    "type": "EMOTION",
                    "metadata": {"valence": -0.7, "arousal": 0.8, "label": "fear"},
                },
                {"id": "s1", "type": "SOMA", "metadata": {"location": "грудь", "sensation": "тяжесть"}},
            ],
            "edges": [
                {"source_node_id": "person:me", "target_node_id": "e1", "relation": "FEELS"},
                {"source_node_id": "e1", "target_node_id": "s1", "relation": "EXPRESSED_AS"},
            ],
        }

    async def generate_live_reply(
        self,
        user_text: str,
        intent: str,
        mood_context: dict | None,
        parts_context: list[dict] | None,
        graph_context: dict | None,
    ) -> str:
        return ""


class BrokenLLMClient:
    async def classify_intent(self, text: str) -> str:
        return "REFLECTION"

    async def extract_all(self, text: str, intent: str, graph_hints: dict | None = None):
        return "{bad_json"

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
        return ""


def test_processor_parses_fenced_llm_json_into_graph(tmp_path):
    async def scenario() -> None:
        db_path = tmp_path / "llm_ok.db"
        api = GraphAPI(GraphStorage(db_path=db_path))
        journal = JournalStorage(db_path=db_path)
        llm_client = FencedLLMClient()
        processor = MessageProcessor(
            graph_api=api,
            journal=journal,
            qdrant=_NoopQdrant(),
            session_memory=SessionMemory(),
            llm_client=llm_client,
            use_llm=True,
        )

        try:
            await processor.process_message(user_id="me", text="тест", source="cli")

            assert llm_client.extract_all_calls == 1
            assert len(await api.get_user_nodes_by_type("me", "PROJECT")) == 1
            assert len(await api.get_user_nodes_by_type("me", "BELIEF")) == 1
            assert len(await api.get_user_nodes_by_type("me", "EMOTION")) == 1
            assert len(await api.get_user_nodes_by_type("me", "SOMA")) == 1
        finally:
            await api.storage.close()

    asyncio.run(scenario())


def test_processor_does_not_crash_on_bad_llm_json(tmp_path):
    async def scenario() -> None:
        db_path = tmp_path / "llm_bad.db"
        api = GraphAPI(GraphStorage(db_path=db_path))
        journal = JournalStorage(db_path=db_path)
        processor = MessageProcessor(
            graph_api=api,
            journal=journal,
            qdrant=_NoopQdrant(),
            session_memory=SessionMemory(),
            llm_client=BrokenLLMClient(),
            use_llm=True,
        )

        try:
            result = await processor.process_message(
                user_id="me",
                text="Я боюсь, что не вывезу проект SELF-OS.",
                source="cli",
            )

            assert result.intent in {"REFLECTION", "FEELING_REPORT"}
            assert len(await api.get_user_nodes_by_type("me", "NOTE")) >= 1
        finally:
            await api.storage.close()

    asyncio.run(scenario())
