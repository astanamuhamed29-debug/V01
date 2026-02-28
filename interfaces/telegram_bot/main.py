from __future__ import annotations

import asyncio
import os

from aiogram import Bot, Dispatcher, F, Router
from aiogram.types import Message
from dotenv import load_dotenv

from core.pipeline.processor import MessageProcessor
from interfaces.processor_factory import build_processor

router = Router()


def _get_bot_token() -> str:
    load_dotenv()
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    if not token:
        raise RuntimeError("TELEGRAM_BOT_TOKEN is not set")
    return token


@router.message(F.text)
async def handle_text_message(message: Message, processor: MessageProcessor) -> None:
    await handle_incoming_message(message, processor)


async def run_bot() -> None:
    token = _get_bot_token()
    bot = Bot(token=token)
    processor = build_processor()
    dispatcher = Dispatcher()
    dispatcher["processor"] = processor
    dispatcher.include_router(router)

    try:
        await dispatcher.start_polling(bot)
    finally:
        await bot.session.close()


def main() -> None:
    asyncio.run(run_bot())


async def handle_incoming_message(message: Message, processor) -> None:
    if message.from_user is None or message.text is None:
        return

    user_id = str(message.from_user.id)
    result = await processor.process(
        user_id,
        message.text,
        source="telegram",
    )

    if result.reply_text:
        await message.answer(result.reply_text)


if __name__ == "__main__":
    main()
