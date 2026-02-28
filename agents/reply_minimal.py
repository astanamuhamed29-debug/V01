from __future__ import annotations

from datetime import datetime

from core.graph.model import Node
from core.search.qdrant_storage import VectorSearchResult


def generate_reply(
    text: str,
    intent: str,
    extracted_structures: dict,
    mood_context: dict | None = None,
    parts_context: list[dict] | None = None,
    graph_context: dict | None = None,
    retrieved_context: list[VectorSearchResult] | None = None,
    session_context: list[dict] | None = None,
    policy: str = "REFLECT",
) -> str:
    nodes: list[Node] = extracted_structures.get("nodes", [])
    edges = extracted_structures.get("edges", [])

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

    history_note = ""
    if graph_context and graph_context.get("has_history"):
        trend = graph_context.get("mood_trend", "unknown")
        recurring = graph_context.get("recurring_emotions", [])
        projects_ctx = graph_context.get("active_projects", [])

        if trend == "declining":
            history_note += " Замечаю что последние несколько дней становится тяжелее."
        elif trend == "improving":
            history_note += " Вижу что в последнее время становится легче."

        for rec in recurring:
            label = rec.get("label", "")
            count = rec.get("count", 0)
            if count >= 3 and label and label not in (trend_note + parts_note):
                history_note += f" {label.capitalize()} встречается уже {count} раз — это важный сигнал."
                break

        if intent == "TASK_LIKE" and projects_ctx:
            proj = projects_ctx[0]
            history_note += f" Связываю это с проектом «{proj}»."

    retrieved_context = retrieved_context or []
    session_context = session_context or []
    system_prompt = (
        "## Терапевтическая тактика\n"
        f"{policy}\n\n"
        "## Похожие ситуации из прошлого пользователя\n"
        f"{[r.payload for r in retrieved_context[:3]]}\n\n"
        "## Контекст текущей сессии\n"
        f"{session_context[-5:]}"
    )
    # NOTE: improved policy-aware context conditioning.

    policy_note = ""
    if policy != "REFLECT":
        policy_note = f" Тактика: {policy}."

    memory_note = ""
    if retrieved_context:
        top = retrieved_context[0]
        memory_note = f" Нашёл похожий паттерн из прошлого (score={top.score:.2f})."

    session_note = ""
    if session_context:
        session_note = " Учитываю контекст текущей сессии."

    conflict_note = ""
    if any(getattr(edge, "relation", "") == "CONFLICTS_WITH" for edge in edges):
        conflict_note = " Вижу внутреннее противоречие между важным для тебя и тем, как сейчас реагирует часть."

    emotions = [node for node in nodes if node.type == "EMOTION"]
    tasks = [node for node in nodes if node.type == "TASK"]
    thoughts = [node for node in nodes if node.type == "THOUGHT"]
    beliefs = [node for node in nodes if node.type == "BELIEF"]
    projects = [node for node in nodes if node.type == "PROJECT"]
    events = [node for node in nodes if node.type == "EVENT"]

    if intent == "META":
        values = [n for n in nodes if n.type == "VALUE"]
        if values:
            val_name = values[0].name or "смысл"
            return f"Слышу запрос на {val_name}. Давай разберём что именно ты ищешь.{history_note}{conflict_note}{policy_note}{memory_note}{session_note}"
        return f"Слышу вопрос о смысле. Что именно хочется получать от этого?{history_note}{conflict_note}{policy_note}{memory_note}{session_note}"

    if intent == "FEELING_REPORT" or emotions:
        emotion_labels = [
            node.metadata.get("label") for node in emotions if node.metadata.get("label")
        ]
        if emotion_labels:
            joined = " и ".join(emotion_labels)
            return f"Слышу: {joined}. Сохранил в граф.{trend_note}{parts_note}{history_note}{conflict_note}{policy_note}{memory_note}{session_note}"
        return f"Слышу тебя. Эмоциональный сигнал записан.{parts_note}{history_note}{conflict_note}{policy_note}{memory_note}{session_note}"

    if intent == "TASK_LIKE" or tasks:
        task_names = [node.text or node.name for node in tasks if node.text or node.name]
        if task_names:
            first_task = task_names[0]
            return f"Принято: «{first_task}». Добавил в SELF-Graph как задачу.{parts_note}{history_note}{conflict_note}{policy_note}{memory_note}{session_note}"
        return f"Принято. Задача добавлена в SELF-Graph.{parts_note}{history_note}{conflict_note}{policy_note}{memory_note}{session_note}"

    if intent == "IDEA":
        idea_nodes = [n for n in nodes if n.type in ("NOTE", "PROJECT", "VALUE")]
        if idea_nodes:
            name = idea_nodes[0].name or "идея"
            return f"Интересная идея: «{name[:80]}». Сохранил в SELF-Graph.{parts_note}{history_note}{conflict_note}{policy_note}{memory_note}{session_note}"
        return f"Записал идею. Хочешь развить?{parts_note}{history_note}{conflict_note}{policy_note}{memory_note}{session_note}"

    if thoughts:
        thought_text = thoughts[0].text or thoughts[0].name or ""
        triggers = [e for e in edges if e.relation == "TRIGGERS" and e.source_node_id == thoughts[0].id]
        if triggers:
            return f"Зафиксировал мысль: «{thought_text[:80]}» и её последствия.{parts_note}{history_note}{conflict_note}{policy_note}{memory_note}{session_note}"
        return f"Зафиксировал мысль: «{thought_text[:80]}».{parts_note}{history_note}{conflict_note}{policy_note}{memory_note}{session_note}"

    if beliefs:
        belief_text = beliefs[0].text or beliefs[0].name or ""
        return f"Зафиксировал убеждение: «{belief_text[:80]}».{parts_note}{history_note}{conflict_note}{policy_note}{memory_note}{session_note}"

    if events:
        event_text = events[0].text or events[0].name or ""
        return f"Записал событие: «{event_text[:80]}».{parts_note}{history_note}{conflict_note}{policy_note}{memory_note}{session_note}"

    if projects:
        project_name = projects[0].name or ""
        return f"Отметил активность по проекту «{project_name}».{parts_note}{history_note}{conflict_note}{policy_note}{memory_note}{session_note}"

    if nodes:
        return f"Записал. Продолжай, я накапливаю структуру твоего SELF-Graph.{parts_note}{history_note}{conflict_note}{policy_note}{memory_note}{session_note}"

    return f"Слышу тебя. Записал в SELF-Graph.{parts_note}{history_note}{conflict_note}{policy_note}{memory_note}{session_note}"
