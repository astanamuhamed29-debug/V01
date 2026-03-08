"""Template-based (LLM-free) reply renderer for the ACT stage.

``render_template_reply`` builds a structured Russian-language response from
extracted graph nodes / edges and contextual signals (mood, parts, history,
policy, retrieved memory).  It is used by :class:`~core.pipeline.stage_act.ActStage`
as the *fallback* path: the LLM live-reply is attempted first; if that call
fails or returns nothing, the template reply is surfaced to the user instead.

The function intentionally has **no I/O side-effects** — it is a pure text
renderer that can be unit-tested without any async infrastructure.
"""

from __future__ import annotations

import logging
from datetime import datetime

from core.graph.model import Node
from core.search.qdrant_storage import VectorSearchResult

logger = logging.getLogger(__name__)


def render_template_reply(
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
    """Render a context-aware template reply without calling the LLM.

    Parameters
    ----------
    text:
        The raw user message (used implicitly via intent / nodes).
    intent:
        Classified intent string (e.g. ``"FEELING_REPORT"``, ``"TASK_LIKE"``).
    extracted_structures:
        Dict with ``"nodes"`` and ``"edges"`` lists produced by the ORIENT stage.
    mood_context:
        Optional mood snapshot dict (valence_avg, dominant_label, sample_count …).
    parts_context:
        Optional list of IFS parts history dicts (part, appearances, last_seen).
    graph_context:
        Optional graph context dict (has_history, mood_trend, recurring_emotions …).
    retrieved_context:
        Optional list of vector-search results for memory recall.
    session_context:
        Optional list of recent session turn dicts.
    policy:
        Therapeutic policy label (default ``"REFLECT"``).

    Returns
    -------
    str
        A single Russian-language reply string.
    """
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
                except (ValueError, TypeError) as exc:
                    logger.debug("Date parse failed: %s", exc)
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
