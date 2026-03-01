"""OODA — OBSERVE stage.

Sanitises input, resolves session, appends to journal, classifies intent.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from uuid import uuid4

from config import MAX_TEXT_LENGTH
from core.context.session_memory import SessionMemory
from core.journal.storage import JournalStorage
from core.pipeline import router
from core.pipeline.events import EventBus

logger = logging.getLogger(__name__)

_CONTROL_CHAR_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def _sanitize_text(text: str) -> str:
    text = _CONTROL_CHAR_RE.sub("", text)
    return text.strip()


@dataclass(slots=True)
class ObserveResult:
    """Output of the OBSERVE stage."""

    text: str
    intent: str
    session_id: str
    timestamp: str


class ObserveStage:
    """Sanitise → journal → session tracking → intent classification."""

    def __init__(
        self,
        journal: JournalStorage,
        session_memory: SessionMemory,
        event_bus: EventBus,
        session_gap: timedelta = timedelta(minutes=30),
    ) -> None:
        self.journal = journal
        self.session_memory = session_memory
        self.event_bus = event_bus
        self._session_gap = session_gap
        self._sessions: dict[str, tuple[str, datetime]] = {}

    async def run(
        self,
        user_id: str,
        raw_text: str,
        *,
        source: str = "cli",
        timestamp: str | None = None,
    ) -> ObserveResult:
        text = _sanitize_text(raw_text)
        if len(text) > MAX_TEXT_LENGTH:
            raise ValueError(f"Message too long: {len(text)} chars (max {MAX_TEXT_LENGTH})")

        ts = timestamp or datetime.now(timezone.utc).isoformat()
        session_id = self._resolve_session_id(user_id)

        await self.journal.append(
            user_id=user_id, timestamp=ts, text=text, source=source,
            session_id=session_id,
        )
        self.event_bus.publish("journal.appended", {"user_id": user_id, "text": text})

        intent = router.classify(text)
        return ObserveResult(text=text, intent=intent, session_id=session_id, timestamp=ts)

    # ------------------------------------------------------------------

    def _resolve_session_id(self, user_id: str) -> str:
        now = datetime.now(timezone.utc)
        prev = self._sessions.get(user_id)
        if prev is None or (now - prev[1]) > self._session_gap:
            sid = str(uuid4())
            self._sessions[user_id] = (sid, now)
            self.session_memory.clear(user_id)
            logger.info("SessionMemory cleared for user=%s (new session %s)", user_id, sid[:8])
            return sid
        self._sessions[user_id] = (prev[0], now)
        return prev[0]
