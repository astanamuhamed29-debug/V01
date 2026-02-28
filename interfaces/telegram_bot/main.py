from __future__ import annotations

import asyncio
import logging
import os
from datetime import datetime, timezone
from pathlib import Path

from aiogram import Bot, Dispatcher, F, Router
from aiogram.filters import Command, CommandStart
from aiogram.types import CallbackQuery, Message
from dotenv import load_dotenv

from core.analytics.calibrator import ThresholdCalibrator
from config import LOG_LEVEL
from core.analytics.pattern_analyzer import PatternAnalyzer
from core.pipeline.processor import MessageProcessor
from core.scheduler.proactive_scheduler import ProactiveScheduler
from interfaces.processor_factory import build_processor

router = Router()
logger = logging.getLogger(__name__)
LOCK_PATH = Path("data/telegram_bot.lock")


class BotInstanceLockError(RuntimeError):
    pass


def _is_process_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def _acquire_bot_instance_lock(lock_path: Path = LOCK_PATH) -> Path:
    lock_path.parent.mkdir(parents=True, exist_ok=True)

    if lock_path.exists():
        try:
            existing_pid = int(lock_path.read_text(encoding="utf-8").strip())
        except (ValueError, OSError):
            existing_pid = 0

        if _is_process_alive(existing_pid):
            raise BotInstanceLockError(
                f"Another local bot instance is already running (pid={existing_pid})."
            )
        lock_path.unlink(missing_ok=True)

    flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY
    fd = os.open(str(lock_path), flags)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        handle.write(str(os.getpid()))

    return lock_path


def _release_bot_instance_lock(lock_path: Path) -> None:
    lock_path.unlink(missing_ok=True)


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


@router.callback_query(F.data.startswith("fb:"))
async def handle_feedback_callback(call: CallbackQuery, processor: MessageProcessor) -> None:
    """Принимает 👍/👎 оценку проактивного сообщения."""
    if call.from_user is None or call.data is None:
        return

    parts = call.data.split(":")
    if len(parts) < 5:
        return

    _, helpful_str, signal_type, score_str, ts_str = parts[:5]
    user_id = str(call.from_user.id)

    try:
        was_helpful = helpful_str == "1"
        score = float(score_str)
        sent_at = datetime.fromtimestamp(int(ts_str), tz=timezone.utc).isoformat()

        await processor.graph_api.storage.save_signal_feedback(
            user_id=user_id,
            signal_type=signal_type,
            signal_score=score,
            was_helpful=was_helpful,
            sent_at=sent_at,
        )

        if hasattr(processor, "calibrator") and processor.calibrator:
            await processor.calibrator.load(user_id)

        reply = "Спасибо, запомню 👌" if was_helpful else "Понял, буду точнее 🙏"
        await call.answer(reply)
        if call.message:
            await call.message.edit_reply_markup(reply_markup=None)
    except Exception as exc:
        logger.warning("feedback callback failed: %s", exc)
        await call.answer("Не смог сохранить оценку")


async def run_bot() -> None:
    token = _get_bot_token()
    bot = Bot(token=token)
    processor = build_processor()
    if not hasattr(processor, "pattern_analyzer"):
        processor.pattern_analyzer = PatternAnalyzer(
            processor.graph_api.storage,
            embedding_service=getattr(processor, "embedding_service", None),
        )
    calibrator = ThresholdCalibrator(processor.graph_api.storage)
    processor.calibrator = calibrator
    scheduler = ProactiveScheduler(
        bot=bot,
        storage=processor.graph_api.storage,
        analyzer=processor.pattern_analyzer,
        calibrator=calibrator,
    )
    dispatcher = Dispatcher()
    dispatcher["processor"] = processor
    dispatcher.include_router(router)

    try:
        scheduler.start()
        await dispatcher.start_polling(bot)
    finally:
        await scheduler.stop()
        await bot.session.close()
        if hasattr(processor.graph_api.storage, "close"):
            await processor.graph_api.storage.close()


def main() -> None:
    logging.basicConfig(level=getattr(logging, LOG_LEVEL, logging.INFO))
    lock_path: Path | None = None
    try:
        lock_path = _acquire_bot_instance_lock()
        asyncio.run(run_bot())
    except BotInstanceLockError as exc:
        logger.error("%s", exc)
        raise SystemExit(1) from exc
    finally:
        if lock_path is not None:
            _release_bot_instance_lock(lock_path)


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
