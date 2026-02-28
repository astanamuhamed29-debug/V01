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

            assert len(notes) == 3
            assert len(projects) == 1
            assert projects[0].name == "SELF-OS"
            assert len(tasks) >= 1
            assert any("набросать архитектуру" in (task.text or "") for task in tasks)
            assert len(beliefs) == 1

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
        )

        try:
            await processor.process(user_id="321", text="Привет, я хочу переехать", source="telegram")

            projects = await api.get_user_nodes_by_type("321", "PROJECT")
            assert any((project.name or "").lower() == "переезд" for project in projects)
        finally:
            await api.storage.close()

    asyncio.run(scenario())
