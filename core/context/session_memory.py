"""Session Memory with sliding-window conversation context for SELF-OS.

Keeps the last *N* messages per user in memory with a configurable TTL so
that the LLM always has recent conversational context without loading the
full journal from disk.

Usage::

    memory = SessionMemory(max_messages=10, ttl_seconds=3600)
    memory.add_message("user42", "Привет!", role="user")
    context = memory.get_context("user42")
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque

__all__ = ["SessionMemory", "SessionMessage"]

_DEFAULT_TTL = 60 * 60  # 1 hour in seconds
_DEFAULT_MAX_MESSAGES = 20


@dataclass
class SessionMessage:
    """A single message stored in :class:`SessionMemory`."""

    role: str
    text: str
    timestamp: float = field(default_factory=time.monotonic)


@dataclass
class _UserSession:
    messages: Deque[SessionMessage]
    last_active: float = field(default_factory=time.monotonic)


class SessionMemory:
    """In-memory sliding-window session storage.

    Parameters
    ----------
    max_messages:
        Maximum number of messages to retain per user.
    ttl_seconds:
        Idle time in seconds after which a user session is expired and
        removed on the next access.
    """

    def __init__(
        self,
        max_messages: int = _DEFAULT_MAX_MESSAGES,
        ttl_seconds: float = _DEFAULT_TTL,
    ) -> None:
        self.max_messages = max_messages
        self.ttl_seconds = ttl_seconds
        self._sessions: dict[str, _UserSession] = {}

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def add_message(
        self,
        user_id: str,
        text: str,
        role: str = "user",
        timestamp: float | None = None,
    ) -> None:
        """Append a message to *user_id*'s session window.

        Parameters
        ----------
        user_id:
            Unique identifier of the user.
        text:
            Message content.
        role:
            Either ``"user"`` or ``"assistant"``.
        timestamp:
            Optional ``time.monotonic()`` timestamp; defaults to now.
        """
        self._evict_if_expired(user_id)
        session = self._get_or_create(user_id)
        msg = SessionMessage(role=role, text=text, timestamp=timestamp or time.monotonic())
        session.messages.append(msg)
        session.last_active = time.monotonic()

    def get_context(
        self,
        user_id: str,
        max_messages: int | None = None,
    ) -> list[dict]:
        """Return the last *max_messages* messages for *user_id*.

        Parameters
        ----------
        user_id:
            User whose context to retrieve.
        max_messages:
            Override the instance-level ``max_messages`` limit.

        Returns
        -------
        list[dict]
            Each dict has keys ``"role"`` and ``"text"``.
        """
        self._evict_if_expired(user_id)
        session = self._sessions.get(user_id)
        if session is None:
            return []
        limit = max_messages if max_messages is not None else self.max_messages
        messages = list(session.messages)[-limit:]
        return [{"role": m.role, "text": m.text} for m in messages]

    def get_summary(self, user_id: str) -> str:
        """Return a brief heuristic summary of the user's session.

        This is a lightweight, LLM-free summary. For a richer summary you
        can pass the output of :meth:`get_context` to an LLM.
        """
        context = self.get_context(user_id)
        if not context:
            return "Нет сообщений в текущей сессии."
        user_messages = [m["text"] for m in context if m["role"] == "user"]
        if not user_messages:
            return "Нет сообщений пользователя в сессии."
        total = len(user_messages)
        preview = user_messages[-1][:80]
        return f"Сессия: {total} сообщение(й). Последнее: «{preview}»"

    def clear(self, user_id: str) -> None:
        """Clear the session for *user_id*."""
        self._sessions.pop(user_id, None)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_or_create(self, user_id: str) -> _UserSession:
        if user_id not in self._sessions:
            self._sessions[user_id] = _UserSession(
                messages=deque(maxlen=self.max_messages)
            )
        return self._sessions[user_id]

    def _evict_if_expired(self, user_id: str) -> None:
        session = self._sessions.get(user_id)
        if session is None:
            return
        if time.monotonic() - session.last_active > self.ttl_seconds:
            del self._sessions[user_id]
