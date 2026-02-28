from __future__ import annotations

from typing import Protocol


class LLMClient(Protocol):
    def classify_intent(self, text: str) -> str: ...

    def extract_semantic(self, text: str, intent: str) -> dict: ...

    def extract_parts(self, text: str, intent: str) -> dict: ...

    def extract_emotion(self, text: str, intent: str) -> dict: ...


class MockLLMClient:
    def classify_intent(self, text: str) -> str:
        lowered = text.lower()
        if any(word in lowered for word in ["надо", "нужно", "сделать"]):
            return "TASK_LIKE"
        if any(word in lowered for word in ["чувств", "боюсь", "страшно"]):
            return "FEELING_REPORT"
        if any(word in lowered for word in ["идея", "придумал"]):
            return "IDEA"
        return "REFLECTION"

    def extract_semantic(self, text: str, intent: str) -> dict:
        return {"nodes": [], "edges": []}

    def extract_parts(self, text: str, intent: str) -> dict:
        return {"nodes": [], "edges": []}

    def extract_emotion(self, text: str, intent: str) -> dict:
        return {"nodes": [], "edges": []}
