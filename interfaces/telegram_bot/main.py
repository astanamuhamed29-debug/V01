from __future__ import annotations

import asyncio
import logging
import os

from aiogram import Bot, Dispatcher, F, Router
from aiogram.filters import Command, CommandStart
from aiogram.types import Message
from dotenv import load_dotenv

from config import LOG_LEVEL
from core.analytics.pattern_analyzer import PatternAnalyzer
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


@router.message(Command("insight"))
async def cmd_insight(message: Message, processor: MessageProcessor) -> None:
    """
    Генерирует текстовый инсайт по накопленным паттернам.
    Использует PatternAnalyzer + LLM для формулировки.
    """
    if message.from_user is None:
        return

    user_id = str(message.from_user.id)
    try:
        analyzer = getattr(processor, "pattern_analyzer", PatternAnalyzer(processor.graph_api.storage))
        report = await analyzer.analyze(user_id, days=30)

        if not report.has_enough_data:
            await message.answer(
                "Пока данных маловато для глубокого анализа.\n"
                "Напиши ещё несколько сообщений — и я начну видеть паттерны."
            )
            return

        insight_lines: list[str] = []

        if report.need_profile:
            top_needs = ", ".join(item.need_name for item in report.need_profile[:3])
            insight_lines.append(f"Топ потребностей: {top_needs}")

        if report.trigger_patterns:
            top_trigger = report.trigger_patterns[0]
            insight_lines.append(
                f"Частый паттерн: «{top_trigger.source_text[:50]}» → "
                f"{top_trigger.target_name} ({top_trigger.occurrences} раз)"
            )

        if report.cognition_patterns:
            top_cog = report.cognition_patterns[0]
            insight_lines.append(
                f"Мыслительный паттерн: {top_cog.distortion_ru} "
                f"({top_cog.count} раз, пример: «{top_cog.example_thought[:40]}»)"
            )

        if report.part_dynamics:
            growing = [part for part in report.part_dynamics if part.trend == "growing"]
            if growing:
                insight_lines.append(f"Активнее становится: {growing[0].part_name}")

        context_text = "\n".join(insight_lines) if insight_lines else "Паттернов пока не обнаружено."

        live_insight = await processor.llm_client.generate_live_reply(
            user_text="/insight",
            intent="META",
            mood_context=None,
            parts_context=None,
            graph_context={
                "has_history": True,
                "insight_data": context_text,
                "is_insight_request": True,
            },
        )

        if live_insight and live_insight.strip():
            await message.answer(live_insight)
        else:
            await message.answer("🔍 Паттерны за последние 30 дней:\n\n" + context_text)
    except Exception as exc:
        logger.warning("insight failed: %s", exc)
        await message.answer("Не смог собрать инсайт. Попробуй позже.")


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
    if not hasattr(processor, "pattern_analyzer"):
        processor.pattern_analyzer = PatternAnalyzer(processor.graph_api.storage)
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
