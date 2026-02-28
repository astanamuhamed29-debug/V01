import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

from interfaces.telegram_bot.main import handle_incoming_message, handle_report_message


class FakeProcessor:
    def __init__(self, reply_text: str) -> None:
        self.reply_text = reply_text
        self.process = AsyncMock(return_value=SimpleNamespace(reply_text=reply_text))
        self.build_weekly_report = AsyncMock(return_value="weekly report")


def test_telegram_handler_routes_message_to_processor_and_replies():
    async def scenario() -> None:
        processor = FakeProcessor(reply_text="Принято")
        message = SimpleNamespace(
            from_user=SimpleNamespace(id=123456),
            text="Привет, я хочу переехать",
            answer=AsyncMock(),
        )

        await handle_incoming_message(message, processor)

        processor.process.assert_awaited_once_with("123456", "Привет, я хочу переехать", source="telegram")
        message.answer.assert_awaited_once_with("Принято")

    asyncio.run(scenario())


def test_telegram_handler_does_not_reply_on_empty_text():
    async def scenario() -> None:
        processor = FakeProcessor(reply_text="")
        message = SimpleNamespace(
            from_user=SimpleNamespace(id=777),
            text="Тест",
            answer=AsyncMock(),
        )

        await handle_incoming_message(message, processor)

        processor.process.assert_awaited_once_with("777", "Тест", source="telegram")
        message.answer.assert_not_awaited()

    asyncio.run(scenario())


def test_telegram_report_command_returns_weekly_report():
    async def scenario() -> None:
        processor = FakeProcessor(reply_text="")
        message = SimpleNamespace(
            from_user=SimpleNamespace(id=999),
            text="/report",
            answer=AsyncMock(),
        )

        await handle_report_message(message, processor)

        processor.build_weekly_report.assert_awaited_once_with("999")
        message.answer.assert_awaited_once_with("weekly report")

    asyncio.run(scenario())
