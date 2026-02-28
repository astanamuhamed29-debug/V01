import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from interfaces.telegram_bot.sender import send_to_user


def test_sender_sends_message_with_int_chat_id_and_closes_session():
    async def scenario() -> None:
        fake_bot = SimpleNamespace(
            send_message=AsyncMock(),
            session=SimpleNamespace(close=AsyncMock()),
        )

        with patch("interfaces.telegram_bot.sender._get_bot_token", return_value="token"), patch(
            "interfaces.telegram_bot.sender.Bot", return_value=fake_bot
        ) as bot_ctor:
            await send_to_user("12345", "hello")

        bot_ctor.assert_called_once_with(token="token")
        fake_bot.send_message.assert_awaited_once_with(chat_id=12345, text="hello")
        fake_bot.session.close.assert_awaited_once()

    asyncio.run(scenario())


def test_sender_closes_session_when_send_fails():
    async def scenario() -> None:
        fake_bot = SimpleNamespace(
            send_message=AsyncMock(side_effect=RuntimeError("boom")),
            session=SimpleNamespace(close=AsyncMock()),
        )

        with patch("interfaces.telegram_bot.sender._get_bot_token", return_value="token"), patch(
            "interfaces.telegram_bot.sender.Bot", return_value=fake_bot
        ):
            with pytest.raises(RuntimeError, match="boom"):
                await send_to_user("777", "payload")

        fake_bot.session.close.assert_awaited_once()

    asyncio.run(scenario())
