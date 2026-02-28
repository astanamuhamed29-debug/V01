from __future__ import annotations

import json
import os
from typing import Any
from typing import Protocol

from dotenv import load_dotenv

from config import LLM_MODEL_ID
from core.llm.prompts import SYSTEM_PROMPT_EXTRACTOR
from core.llm.reply_prompt import REPLY_SYSTEM_PROMPT


import logging

logger = logging.getLogger(__name__)


class LLMClient(Protocol):
    async def classify_intent(self, text: str) -> str: ...

    async def extract_all(self, text: str, intent: str, graph_hints: dict | None = None) -> dict[str, Any] | str: ...

    async def extract_semantic(self, text: str, intent: str) -> dict[str, Any] | str: ...

    async def extract_parts(self, text: str, intent: str) -> dict[str, Any] | str: ...

    async def extract_emotion(self, text: str, intent: str) -> dict[str, Any] | str: ...

    async def generate_live_reply(
        self,
        user_text: str,
        intent: str,
        mood_context: dict | None,
        parts_context: list[dict] | None,
        graph_context: dict | None,
    ) -> str: ...


class MockLLMClient:
    async def classify_intent(self, text: str) -> str:
        lowered = text.lower()
        if any(word in lowered for word in ["надо", "нужно", "сделать", "сделай", "слеоай"]):
            return "TASK_LIKE"
        if any(word in lowered for word in ["чувств", "боюсь", "страшно"]):
            return "FEELING_REPORT"
        if any(word in lowered for word in ["идея", "придумал"]):
            return "IDEA"
        return "REFLECTION"

    async def extract_all(self, text: str, intent: str, graph_hints: dict | None = None) -> dict[str, Any]:
        return {"nodes": [], "edges": []}

    async def extract_semantic(self, text: str, intent: str) -> dict[str, Any]:
        return {"nodes": [], "edges": []}

    async def extract_parts(self, text: str, intent: str) -> dict[str, Any]:
        return {"nodes": [], "edges": []}

    async def extract_emotion(self, text: str, intent: str) -> dict[str, Any]:
        return {"nodes": [], "edges": []}

    async def generate_live_reply(
        self,
        user_text: str,
        intent: str,
        mood_context: dict | None,
        parts_context: list[dict] | None,
        graph_context: dict | None,
    ) -> str:
        return ""


class OpenRouterQwenClient:
    def __init__(self, *, api_key: str | None = None, model_id: str | None = None) -> None:
        load_dotenv()
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY", "")
        self.model_id = model_id or os.getenv("OPENROUTER_MODEL_ID", LLM_MODEL_ID)
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

    async def extract_all(self, text: str, intent: str, graph_hints: dict | None = None) -> dict[str, Any] | str:
        payload = {
            "task": "extract_all",
            "text": text,
        }
        if graph_hints:
            if graph_hints.get("known_projects"):
                payload["known_projects"] = graph_hints["known_projects"]
            if graph_hints.get("known_parts"):
                payload["known_parts"] = graph_hints["known_parts"]
            if graph_hints.get("known_values"):
                payload["known_values"] = graph_hints["known_values"]
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

    async def generate_live_reply(
        self,
        user_text: str,
        intent: str,
        mood_context: dict | None,
        parts_context: list[dict] | None,
        graph_context: dict | None,
    ) -> str:
        context_lines: list[str] = []

        if graph_context and graph_context.get("has_history"):
            trend = graph_context.get("mood_trend", "unknown")
            if trend == "declining":
                context_lines.append("Тренд последних дней: становится тяжелее")
            elif trend == "improving":
                context_lines.append("Тренд последних дней: становится легче")

            recurring = graph_context.get("recurring_emotions", [])
            if recurring:
                top = recurring[0]
                context_lines.append(
                    f"Повторяющееся состояние: {top.get('label', '')} ({top.get('count', 0)} раз)"
                )

            values = graph_context.get("known_values", [])
            if values:
                val_names = ", ".join(v.get("name", "") for v in values[:2] if v.get("name"))
                if val_names:
                    context_lines.append(f"Важные ценности: {val_names}")

        if graph_context and graph_context.get("session_conflict"):
            context_lines.append("Обнаружено внутреннее противоречие между ценностью и активной частью")

        if mood_context:
            label = mood_context.get("dominant_label")
            d = mood_context.get("dominance_avg", 0)
            if label:
                feel_str = f"Текущее состояние: {label}"
                if d < -0.3:
                    feel_str += " (ощущение потери контроля)"
                elif d > 0.3:
                    feel_str += " (ощущение контроля)"
                context_lines.append(feel_str)

        if parts_context:
            for ph in parts_context:
                if not ph.get("part"):
                    continue
                part = ph["part"]
                appearances = ph.get("appearances", 1)
                name = part.name or part.subtype or ""
                voice = part.metadata.get("voice", "")
                line = f"Активная часть: {name}"
                if appearances > 1:
                    line += f" (появляется {appearances}-й раз)"
                if voice:
                    line += f", голос: «{voice}»"
                context_lines.append(line)

        context_block = "\n".join(context_lines) if context_lines else "Первое обращение пользователя."

        user_payload = (
            f"Intent: {intent}\n"
            f"Сообщение: {user_text}\n\n"
            f"Контекст:\n{context_block}\n\n"
            "Ответь пользователю."
        )

        client = self._get_client()
        if client is None:
            return ""

        try:
            completion = await client.chat.completions.create(
                model=self.model_id,
                temperature=0.7,
                max_tokens=300,
                messages=[
                    {
                        "role": "system",
                        "content": REPLY_SYSTEM_PROMPT,
                        "cache_control": {"type": "ephemeral"},
                    },
                    {
                        "role": "user",
                        "content": user_payload,
                    },
                ],
            )
        except Exception as exc:
            logger.error("generate_live_reply failed: %s", exc)
            return ""

        usage = getattr(completion, "usage", None)
        if usage:
            logger.info(
                "LiveReply tokens: prompt=%s completion=%s total=%s",
                getattr(usage, "prompt_tokens", "?"),
                getattr(usage, "completion_tokens", "?"),
                getattr(usage, "total_tokens", "?"),
            )

        if not completion.choices:
            return ""

        reply = completion.choices[0].message.content or ""
        return reply.strip()

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
                max_tokens=2000,
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
