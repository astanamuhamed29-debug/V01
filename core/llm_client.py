from __future__ import annotations

import json
import os
from typing import Any
from typing import Protocol

from dotenv import load_dotenv

from config import LLM_MODEL_ID
from core.llm.prompts import SYSTEM_PROMPT_EXTRACTOR


import logging

logger = logging.getLogger(__name__)


class LLMClient(Protocol):
    async def classify_intent(self, text: str) -> str: ...

    async def extract_all(self, text: str, intent: str) -> dict[str, Any] | str: ...

    async def extract_semantic(self, text: str, intent: str) -> dict[str, Any] | str: ...

    async def extract_parts(self, text: str, intent: str) -> dict[str, Any] | str: ...

    async def extract_emotion(self, text: str, intent: str) -> dict[str, Any] | str: ...


class MockLLMClient:
    async def classify_intent(self, text: str) -> str:
        lowered = text.lower()
        if any(word in lowered for word in ["надо", "нужно", "сделать"]):
            return "TASK_LIKE"
        if any(word in lowered for word in ["чувств", "боюсь", "страшно"]):
            return "FEELING_REPORT"
        if any(word in lowered for word in ["идея", "придумал"]):
            return "IDEA"
        return "REFLECTION"

    async def extract_all(self, text: str, intent: str) -> dict[str, Any]:
        return {"nodes": [], "edges": []}

    async def extract_semantic(self, text: str, intent: str) -> dict[str, Any]:
        return {"nodes": [], "edges": []}

    async def extract_parts(self, text: str, intent: str) -> dict[str, Any]:
        return {"nodes": [], "edges": []}

    async def extract_emotion(self, text: str, intent: str) -> dict[str, Any]:
        return {"nodes": [], "edges": []}


class OpenRouterQwenClient:
    def __init__(self, *, api_key: str | None = None, model_id: str | None = None) -> None:
        load_dotenv()
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY", "")
        self.model_id = model_id or LLM_MODEL_ID
        self.base_url = "https://openrouter.ai/api/v1"

        self._client: Any | None = None
        self._client_init_error: str | None = None

    async def classify_intent(self, text: str) -> str:
        payload = {
            "task": "intent_classification",
            "intent_set": [
                "REFLECTION",
                "EVENT_REPORT",
                "IDEA",
                "TASK_LIKE",
                "FEELING_REPORT",
                "META",
            ],
            "text": text,
        }
        response = await self._chat_json(payload)
        if not response:
            return "REFLECTION"

        try:
            data = json.loads(response)
            intent = str(data.get("intent", "REFLECTION")).upper()
            if intent in {"REFLECTION", "EVENT_REPORT", "IDEA", "TASK_LIKE", "FEELING_REPORT", "META"}:
                return intent
        except json.JSONDecodeError:
            pass
        return "REFLECTION"

    async def extract_all(self, text: str, intent: str) -> dict[str, Any] | str:
        payload = {
            "task": "extract_all",
            "text": text,
        }
        raw = await self._chat_json(payload)
        if not raw:
            return {"nodes": [], "edges": []}
        return raw

    async def extract_semantic(self, text: str, intent: str) -> dict[str, Any] | str:
        return await self._extract_by_scope(text=text, intent=intent, scope="semantic")

    async def extract_parts(self, text: str, intent: str) -> dict[str, Any] | str:
        return await self._extract_by_scope(text=text, intent=intent, scope="parts")

    async def extract_emotion(self, text: str, intent: str) -> dict[str, Any] | str:
        return await self._extract_by_scope(text=text, intent=intent, scope="emotion")

    async def _extract_by_scope(self, *, text: str, intent: str, scope: str) -> dict[str, Any] | str:
        payload = {
            "task": "extract_graph",
            "scope": scope,
            "intent_hint": intent,
            "text": text,
        }
        raw = await self._chat_json(payload)
        if not raw:
            return {"nodes": [], "edges": []}
        return raw

    async def _chat_json(self, payload: dict[str, Any]) -> str:
        client = self._get_client()
        if client is None:
            logger.warning("LLM client unavailable, skipping API call")
            return ""

        try:
            completion = await client.chat.completions.create(
                model=self.model_id,
                temperature=0,
                messages=[
                    {
                        "role": "system",
                        "content": SYSTEM_PROMPT_EXTRACTOR,
                        "cache_control": {"type": "ephemeral"},
                    },
                    {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
                ],
            )
        except Exception:
            return ""

        usage = getattr(completion, "usage", None)
        if usage:
            logger.info(
                "LLM tokens: prompt=%s completion=%s total=%s",
                getattr(usage, "prompt_tokens", "?"),
                getattr(usage, "completion_tokens", "?"),
                getattr(usage, "total_tokens", "?"),
            )

        if not completion.choices:
            return ""

        content = completion.choices[0].message.content
        return content or ""

    def _get_client(self):
        if self._client is not None:
            return self._client

        if not self.api_key:
            self._client_init_error = "OPENROUTER_API_KEY is not set"
            logger.error("OpenRouterQwenClient: OPENROUTER_API_KEY is not set — LLM disabled")
            return None

        try:
            from openai import AsyncOpenAI
        except Exception:
            self._client_init_error = "openai package is not installed"
            logger.error("OpenRouterQwenClient: openai package missing — LLM disabled")
            return None

        self._client = AsyncOpenAI(api_key=self.api_key, base_url=self.base_url)
        return self._client
