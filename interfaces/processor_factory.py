from __future__ import annotations

import os

from config import DB_PATH as _DEFAULT_DB_PATH
from core.graph.api import GraphAPI
from core.graph.storage import GraphStorage
from core.journal.storage import JournalStorage
from core.llm_client import OpenRouterQwenClient
from core.pipeline.processor import MessageProcessor


def build_processor(db_path: str | None = None) -> MessageProcessor:
    resolved = db_path or _DEFAULT_DB_PATH
    graph_storage = GraphStorage(db_path=resolved)
    graph_api = GraphAPI(graph_storage)
    journal_storage = JournalStorage(db_path=resolved)
    llm_client = OpenRouterQwenClient()
    return MessageProcessor(graph_api=graph_api, journal=journal_storage, llm_client=llm_client)
