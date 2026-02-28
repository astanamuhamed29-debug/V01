import asyncio

from core.graph.api import GraphAPI
from core.graph.storage import GraphStorage
from core.journal.storage import JournalStorage
from core.pipeline.processor import MessageProcessor


def test_part_value_conflict_edge_is_created(tmp_path):
    async def scenario() -> None:
        db_path = tmp_path / "conflicts.db"
        storage = GraphStorage(db_path=db_path)
        api = GraphAPI(storage)
        journal = JournalStorage(db_path=db_path)
        processor = MessageProcessor(graph_api=api, journal=journal, use_llm=False)

        try:
            text = "Хочу сделать вывод более живым, но мне это не нравится"
            result = await processor.process_message(user_id="u1", text=text, source="cli")

            assert any(node.type == "VALUE" for node in result.nodes)
            assert any(node.type == "PART" for node in result.nodes)
            assert any(edge.relation == "CONFLICTS_WITH" for edge in result.edges)
        finally:
            await storage.close()

    asyncio.run(scenario())
