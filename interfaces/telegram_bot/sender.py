from __future__ import annotations

import os

from aiogram import Bot
from dotenv import load_dotenv


def _get_bot_token() -> str:
    load_dotenv()
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    if not token:
        raise RuntimeError("TELEGRAM_BOT_TOKEN is not set")
    return token


async def send_to_user(user_id: str, text: str) -> None:
    bot = Bot(token=_get_bot_token())
    try:
        await bot.send_message(chat_id=user_id, text=text)
    finally:
        await bot.session.close()
