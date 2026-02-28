from __future__ import annotations

import os

from aiogram import Bot, Dispatcher, F
from aiogram.types import Message
from dotenv import load_dotenv

from interfaces.cli.main import build_processor


def _get_bot_token() -> str:
    load_dotenv()
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    if not token:
        raise RuntimeError("TELEGRAM_BOT_TOKEN is not set")
    return token


async def run_bot() -> None:
    token = _get_bot_token()
    bot = Bot(token=token)
    dispatcher = Dispatcher()
    processor = build_processor()

    @dispatcher.message(F.text)
    async def handle_text_message(message: Message) -> None:
        await handle_incoming_message(message, processor)

    try:
        await dispatcher.start_polling(bot)
    finally:
        await bot.session.close()


def main() -> None:
    import asyncio

    asyncio.run(run_bot())


if __name__ == "__main__":
    main()


async def handle_incoming_message(message: Message, processor) -> None:
    if message.from_user is None or message.text is None:
        return

    user_id = str(message.from_user.id)
    result = await processor.process_message(
        user_id,
        message.text,
        source="telegram",
    )

    if result.reply_text:
        await message.answer(result.reply_text)
