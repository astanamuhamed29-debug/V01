from __future__ import annotations

from core.graph.model import Node


def generate_reply(text: str, intent: str, extracted_structures: dict, mood_context: dict | None = None) -> str:
    nodes: list[Node] = extracted_structures.get("nodes", [])

    trend_note = ""
    if mood_context:
        v = mood_context.get("valence_avg", 0)
        label = mood_context.get("dominant_label", "")
        count = mood_context.get("sample_count", 1)
        if v < -0.5 and count >= 3:
            trend_note = f" Замечаю, что {label or 'негативное состояние'} повторяется — это уже {count}-й раз подряд."
        elif v < -0.3:
            trend_note = " Фиксирую нарастающее напряжение."

    emotions = [node for node in nodes if node.type == "EMOTION"]
    tasks = [node for node in nodes if node.type == "TASK"]
    beliefs = [node for node in nodes if node.type == "BELIEF"]
    projects = [node for node in nodes if node.type == "PROJECT"]
    events = [node for node in nodes if node.type == "EVENT"]

    if intent == "FEELING_REPORT" or emotions:
        emotion_labels = [
            node.metadata.get("label") for node in emotions if node.metadata.get("label")
        ]
        if emotion_labels:
            joined = " и ".join(emotion_labels)
            return f"Слышу: {joined}. Сохранил в граф.{trend_note}"
        return "Слышу тебя. Эмоциональный сигнал записан."

    if intent == "TASK_LIKE" or tasks:
        task_names = [node.text or node.name for node in tasks if node.text or node.name]
        if task_names:
            first_task = task_names[0]
            return f"Принято: «{first_task}». Добавил в SELF-Graph как задачу."
        return "Принято. Задача добавлена в SELF-Graph."

    if beliefs:
        belief_text = beliefs[0].text or beliefs[0].name or ""
        return f"Зафиксировал убеждение: «{belief_text[:80]}»."

    if events:
        event_text = events[0].text or events[0].name or ""
        return f"Записал событие: «{event_text[:80]}»."

    if projects:
        project_name = projects[0].name or ""
        return f"Отметил активность по проекту «{project_name}»."

    if nodes:
        return "Записал. Продолжай, я накапливаю структуру твоего SELF-Graph."

    return ""
