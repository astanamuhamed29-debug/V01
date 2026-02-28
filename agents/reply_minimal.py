from __future__ import annotations

from datetime import datetime

from core.graph.model import Node


def generate_reply(
    text: str,
    intent: str,
    extracted_structures: dict,
    mood_context: dict | None = None,
    parts_context: list[dict] | None = None,
) -> str:
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

    parts_note = ""
    if parts_context:
        for ph in parts_context:
            if not ph.get("part"):
                continue
            part = ph["part"]
            appearances = ph["appearances"]
            last_seen_raw = ph.get("last_seen") or ""
            name = part.name or part.subtype or "часть"
            voice = part.metadata.get("voice", "")

            if appearances > 1:
                try:
                    dt = datetime.fromisoformat(last_seen_raw.replace("Z", ""))
                    months_ru = {
                        1: "января",
                        2: "февраля",
                        3: "марта",
                        4: "апреля",
                        5: "мая",
                        6: "июня",
                        7: "июля",
                        8: "августа",
                        9: "сентября",
                        10: "октября",
                        11: "ноября",
                        12: "декабря",
                    }
                    date_str = f"{dt.day} {months_ru.get(dt.month, '')}".strip()
                except Exception:
                    date_str = "ранее"
                parts_note += f" Замечаю {name} — он появляется уже {appearances}-й раз (последний раз {date_str})."
            else:
                parts_note += f" Замечаю {name}."

            if voice:
                parts_note += f" Его голос: «{voice}»"

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
            return f"Слышу: {joined}. Сохранил в граф.{trend_note}{parts_note}"
        return f"Слышу тебя. Эмоциональный сигнал записан.{parts_note}"

    if intent == "TASK_LIKE" or tasks:
        task_names = [node.text or node.name for node in tasks if node.text or node.name]
        if task_names:
            first_task = task_names[0]
            return f"Принято: «{first_task}». Добавил в SELF-Graph как задачу.{parts_note}"
        return f"Принято. Задача добавлена в SELF-Graph.{parts_note}"

    if beliefs:
        belief_text = beliefs[0].text or beliefs[0].name or ""
        return f"Зафиксировал убеждение: «{belief_text[:80]}».{parts_note}"

    if events:
        event_text = events[0].text or events[0].name or ""
        return f"Записал событие: «{event_text[:80]}».{parts_note}"

    if projects:
        project_name = projects[0].name or ""
        return f"Отметил активность по проекту «{project_name}».{parts_note}"

    if nodes:
        return f"Записал. Продолжай, я накапливаю структуру твоего SELF-Graph.{parts_note}"

    return ""
