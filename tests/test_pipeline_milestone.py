"""Milestone-тесты: проверяем что LLM-экстракция корректно маппится в граф.

Экстракция мокается — тестируем именно orient→decide→act пайплайн.
"""

import asyncio
import json

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


class _MilestoneLLM:
    """LLM-мок: возвращает разные ноды в зависимости от текста."""

    async def classify_intent(self, text):
        return "REFLECTION"

    async def extract_all(self, text, intent, graph_hints=None):
        lowered = text.lower()
        if "свою личную ос" in lowered or "self-os" in lowered:
            return json.dumps({
                "_reasoning": {"situation": "Хочет создать SELF-OS", "appraisal": "", "affect": "",
                               "defenses": "", "core_needs": ""},
                "intent": "IDEA",
                "nodes": [
                    {"id": "n1", "type": "PROJECT", "name": "SELF-OS",
                     "key": "project:self-os", "metadata": {}},
                ],
                "edges": [
                    {"source_node_id": "person:me", "target_node_id": "n1",
                     "relation": "OWNS_PROJECT"},
                ]
            }, ensure_ascii=False)
        elif "набросать архитектуру" in lowered:
            return json.dumps({
                "_reasoning": {"situation": "Задача", "appraisal": "", "affect": "",
                               "defenses": "", "core_needs": ""},
                "intent": "TASK_LIKE",
                "nodes": [
                    {"id": "n1", "type": "TASK", "text": "набросать архитектуру",
                     "key": "task:набросать архитектуру", "metadata": {}},
                ],
                "edges": [
                    {"source_node_id": "person:me", "target_node_id": "n1",
                     "relation": "HAS_TASK"},
                ]
            }, ensure_ascii=False)
        elif "боюсь" in lowered:
            return json.dumps({
                "_reasoning": {"situation": "Страх", "appraisal": "Предсказание провала",
                               "affect": "Страх", "defenses": "", "core_needs": "Безопасность"},
                "intent": "FEELING_REPORT",
                "nodes": [
                    {"id": "n1", "type": "BELIEF", "text": "боюсь не вывезти",
                     "key": "belief:боюсь не вывезти", "metadata": {}},
                    {"id": "n2", "type": "EMOTION",
                     "metadata": {"label": "страх", "valence": -0.8, "arousal": 0.6,
                                  "dominance": -0.6, "intensity": 0.9}},
                ],
                "edges": [
                    {"source_node_id": "person:me", "target_node_id": "n1",
                     "relation": "HOLDS_BELIEF"},
                    {"source_node_id": "person:me", "target_node_id": "n2",
                     "relation": "FEELS"},
                ]
            }, ensure_ascii=False)
        elif "переехать" in lowered:
            return json.dumps({
                "_reasoning": {"situation": "Переезд", "appraisal": "", "affect": "",
                               "defenses": "", "core_needs": ""},
                "intent": "IDEA",
                "nodes": [
                    {"id": "n1", "type": "PROJECT", "name": "переезд",
                     "key": "project:переезд", "metadata": {}},
                ],
                "edges": [
                    {"source_node_id": "person:me", "target_node_id": "n1",
                     "relation": "OWNS_PROJECT"},
                ]
            }, ensure_ascii=False)
        return json.dumps({"intent": "REFLECTION", "nodes": [], "edges": []},
                          ensure_ascii=False)

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


def test_milestone_scenario_builds_expected_graph(tmp_path):
    async def scenario() -> None:
        db_path = tmp_path / "self_os.db"
        api = GraphAPI(GraphStorage(db_path=db_path))
        journal = JournalStorage(db_path=db_path)
        processor = MessageProcessor(
            graph_api=api,
            journal=journal,
            qdrant=_NoopQdrant(),
            session_memory=SessionMemory(),
            llm_client=_MilestoneLLM(),
        )

        try:
            messages = [
                "Хочу сделать свою личную ОС SELF-OS.",
                "Надо выделить вечер, чтобы набросать архитектуру.",
                "Я боюсь, что не вывезу такой большой проект.",
            ]

            for text in messages:
                await processor.process_message(user_id="me", text=text, source="cli")

            notes = await api.get_user_nodes_by_type("me", "NOTE")
            projects = await api.get_user_nodes_by_type("me", "PROJECT")
            tasks = await api.get_user_nodes_by_type("me", "TASK")
            beliefs = await api.get_user_nodes_by_type("me", "BELIEF")

            # LLM-экстракция создала нужные структуры
            assert len(projects) == 1
            assert projects[0].name == "SELF-OS"
            assert len(tasks) >= 1
            assert any("набросать архитектуру" in (task.text or "") for task in tasks)
            assert len(beliefs) >= 1

            edges = await api.storage.list_edges("me")
            relations = {edge.relation for edge in edges}
            assert "OWNS_PROJECT" in relations
            assert "HAS_TASK" in relations
            assert "HOLDS_BELIEF" in relations
        finally:
            await api.storage.close()

    asyncio.run(scenario())


def test_relocation_phrase_creates_relocation_project(tmp_path):
    async def scenario() -> None:
        db_path = tmp_path / "self_os.db"
        api = GraphAPI(GraphStorage(db_path=db_path))
        journal = JournalStorage(db_path=db_path)
        processor = MessageProcessor(
            graph_api=api,
            journal=journal,
            qdrant=_NoopQdrant(),
            session_memory=SessionMemory(),
            llm_client=_MilestoneLLM(),
        )

        try:
            await processor.process(user_id="321", text="Привет, я хочу переехать", source="telegram")

            projects = await api.get_user_nodes_by_type("321", "PROJECT")
            assert any((project.name or "").lower() == "переезд" for project in projects)
        finally:
            await api.storage.close()

    asyncio.run(scenario())
