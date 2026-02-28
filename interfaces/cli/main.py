from __future__ import annotations

import asyncio

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


async def run_cli() -> None:
    processor = build_processor()
    user_id = "me"
    print("SELF-OS CLI (Stage 1). Введите текст, 'exit' для выхода.")

    while True:
        text = input("> ").strip()
        if text.lower() in {"exit", "quit", "q"}:
            print("Пока.")
            break
        if not text:
            continue

        result = await processor.process_message(user_id=user_id, text=text, source="cli")
        print(f"[intent={result.intent}]")
        if result.reply_text:
            print(result.reply_text)


if __name__ == "__main__":
    asyncio.run(run_cli())
