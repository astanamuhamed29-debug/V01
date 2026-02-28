from __future__ import annotations

import asyncio
import logging
import os

from aiogram import Bot, Dispatcher, F, Router
from aiogram.filters import Command, CommandStart
from aiogram.types import Message
from dotenv import load_dotenv

from config import LOG_LEVEL
from core.pipeline.processor import MessageProcessor
from interfaces.processor_factory import build_processor

router = Router()
logger = logging.getLogger(__name__)


def _get_bot_token() -> str:
    load_dotenv()
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    if not token:
        raise RuntimeError("TELEGRAM_BOT_TOKEN is not set")
    return token


@router.message(Command("report"))
async def handle_report_message(message: Message, processor: MessageProcessor) -> None:
    if message.from_user is None:
        return
    user_id = str(message.from_user.id)
    try:
        report = await processor.build_weekly_report(user_id)
        await message.answer(report)
    except Exception:
        logger.exception("Telegram /report failed for user=%s", user_id)
        await message.answer("Не смог собрать отчёт прямо сейчас. Попробуй ещё раз.")


@router.message(CommandStart())
async def cmd_start(message: Message) -> None:
    await message.answer(
        "Привет. Я SELF-OS.\n\n"
        "Пиши мне всё что думаешь, чувствуешь, планируешь.\n"
        "На любом языке — русском, английском, вперемешку.\n\n"
        "Я буду слушать, замечать паттерны\n"
        "и отражать что происходит внутри.\n\n"
        "Начни прямо сейчас."
    )


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
        if hasattr(processor.graph_api.storage, "close"):
            await processor.graph_api.storage.close()


def main() -> None:
    logging.basicConfig(level=getattr(logging, LOG_LEVEL, logging.INFO))
    asyncio.run(run_bot())


async def handle_incoming_message(message: Message, processor) -> None:
    if message.from_user is None or message.text is None:
        return

    user_id = str(message.from_user.id)
    try:
        logger.info("Telegram message received from user=%s", user_id)
        result = await processor.process(
            user_id,
            message.text,
            source="telegram",
        )

        if result.reply_text:
            await message.answer(result.reply_text)
            logger.info("Telegram reply sent to user=%s", user_id)
    except Exception:
        logger.exception("Telegram handler failed for user=%s", user_id)
        await message.answer("Поймал ошибку при обработке сообщения. Попробуй ещё раз.")


if __name__ == "__main__":
    main()
