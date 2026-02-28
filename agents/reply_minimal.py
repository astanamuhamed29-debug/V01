from __future__ import annotations


def generate_reply(text: str, intent: str, extracted_structures: dict) -> str:
    node_count = len(extracted_structures.get("nodes", []))

    if intent == "TASK_LIKE":
        return "Принято. Я добавил это как рабочее действие в твой SELF-Graph."
    if intent == "FEELING_REPORT":
        return "Слышу тебя. Я сохранил это как важный эмоциональный сигнал."
    if node_count > 0:
        return "Записал. Продолжай, я накапливаю структуру твоего SELF-Graph."
    return ""
