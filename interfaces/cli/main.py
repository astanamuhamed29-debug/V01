from __future__ import annotations

import asyncio

from interfaces.processor_factory import build_processor


async def run_cli() -> None:
    processor = build_processor()  # background_mode=True by default
    user_id = "me"
    print("SELF-OS CLI (Stage 1). Введите текст, 'exit' для выхода.")

    while True:
        text = input("> ").strip()
        if text.lower() in {"exit", "quit", "q"}:
            # Дождаться завершения фоновых задач перед выходом
            await processor.flush_pending()
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
