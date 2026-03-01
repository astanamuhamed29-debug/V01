"""Тест обнаружения конфликтов PART↔VALUE на стадии DECIDE.

Экстракция здесь мокается — мы подаём готовые ноды в граф,
проверяем что DECIDE-стадия корректно создаёт CONFLICTS_WITH ребро.
"""

import asyncio

from core.context.session_memory import SessionMemory
from core.graph.api import GraphAPI
from core.graph.model import Node, Edge
from core.graph.storage import GraphStorage
from core.journal.storage import JournalStorage
from core.pipeline.processor import MessageProcessor


class _NoopQdrant:
    def upsert_embeddings_batch(self, points):
        return

    def search_similar(self, *args, **kwargs):
        return []


class _MockLLMWithConflict:
    """LLM-мок: возвращает VALUE + PART, чтобы DECIDE мог найти конфликт."""

    async def classify_intent(self, text):
        return "REFLECTION"

    async def extract_all(self, text, intent, graph_hints=None):
        import json
        return json.dumps({
            "_reasoning": {
                "situation": "Хочет оживить вывод, но внутренняя часть сопротивляется",
                "appraisal": "Конфликт между ценностью и критиком",
                "affect": "Раздражение",
                "defenses": "Критик блокирует",
                "core_needs": "Самовыражение"
            },
            "intent": "REFLECTION",
            "nodes": [
                {"id": "n1", "type": "VALUE", "name": "живость", "key": "value:живость",
                 "text": "хочу сделать вывод более живым", "metadata": {}},
                {"id": "n2", "type": "PART", "subtype": "critic", "name": "Критик",
                 "key": "part:critic", "text": "мне это не нравится",
                 "metadata": {"voice": "Не нравится, не связывайся"}},
            ],
            "edges": [
                {"source_node_id": "person:me", "target_node_id": "n1", "relation": "HAS_VALUE"},
                {"source_node_id": "person:me", "target_node_id": "n2", "relation": "HAS_PART"},
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


def test_part_value_conflict_edge_is_created(tmp_path):
    async def scenario() -> None:
        db_path = tmp_path / "conflicts.db"
        storage = GraphStorage(db_path=db_path)
        api = GraphAPI(storage)
        journal = JournalStorage(db_path=db_path)
        processor = MessageProcessor(
            graph_api=api,
            journal=journal,
            qdrant=_NoopQdrant(),
            session_memory=SessionMemory(),
            llm_client=_MockLLMWithConflict(),
        )

        try:
            text = "Хочу сделать вывод более живым, но мне это не нравится"
            result = await processor.process_message(user_id="u1", text=text, source="cli")

            assert any(node.type == "VALUE" for node in result.nodes)
            assert any(node.type == "PART" for node in result.nodes)
            assert any(edge.relation == "CONFLICTS_WITH" for edge in result.edges)
        finally:
            await storage.close()

    asyncio.run(scenario())
