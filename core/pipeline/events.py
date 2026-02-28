from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any


@dataclass(slots=True)
class Event:
    name: str
    payload: dict[str, Any]
    timestamp: str


class EventBus:
    def __init__(self) -> None:
        self._subscribers: dict[str, list[Callable[[Event], None]]] = defaultdict(list)

    def subscribe(self, event_name: str, handler: Callable[[Event], None]) -> None:
        self._subscribers[event_name].append(handler)

    def publish(self, event_name: str, payload: dict[str, Any]) -> None:
        event = Event(
            name=event_name,
            payload=payload,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )
        for handler in self._subscribers.get(event_name, []):
            handler(event)
