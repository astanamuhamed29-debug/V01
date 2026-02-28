from __future__ import annotations

from core.graph.api import GraphAPI
from core.graph.storage import GraphStorage
from core.journal.storage import JournalStorage
from core.llm_client import OpenRouterQwenClient
from core.pipeline.processor import MessageProcessor


def build_processor(db_path: str = "data/self_os.db") -> MessageProcessor:
    graph_storage = GraphStorage(db_path=db_path)
    graph_api = GraphAPI(graph_storage)
    journal_storage = JournalStorage(db_path=db_path)
    llm_client = OpenRouterQwenClient()
    return MessageProcessor(graph_api=graph_api, journal=journal_storage, llm_client=llm_client)
