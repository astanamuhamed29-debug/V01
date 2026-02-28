from __future__ import annotations

import re

from core.graph.model import Edge, Node


PART_RULES = [
    {
        "subtype": "critic",
        "name": "Критик",
        "key": "part:critic",
        "patterns": [r"ненавижу себя", r"я идиот", r"снова не сделал", r"всегда так", r"подв[её]л", r"не нравится"],
        "voice": "Ты снова подвёл.",
        "default_text": "Ненавижу себя за это",
    },
    {
        "subtype": "protector",
        "name": "Защитник",
        "key": "part:protector",
        "patterns": [r"не могу начать", r"избег", r"прокрастинац", r"откладыва"],
        "voice": "Я пытаюсь защитить тебя от боли и провала.",
        "default_text": "Не могу начать",
    },
    {
        "subtype": "exile",
        "name": "Изгнанник",
        "key": "part:exile",
        "patterns": [r"стыд", r"страх", r"никто не пойм[её]т", r"старая боль", r"отверж"],
        "voice": "Мне очень больно и стыдно.",
        "default_text": "Мне больно и страшно",
    },
    {
        "subtype": "manager",
        "name": "Менеджер",
        "key": "part:manager",
        "patterns": [r"надо вс[её] успеть", r"списки", r"контрол", r"тревога о будущем"],
        "voice": "Нужно всё держать под контролем.",
        "default_text": "Надо всё успеть",
    },
    {
        "subtype": "firefighter",
        "name": "Пожарный",
        "key": "part:firefighter",
        "patterns": [r"залип в игр", r"залип", r"переел", r"запой", r"соцсети"],
        "voice": "Мне нужно было сбежать от напряжения",
        "default_text": "залип в игры вместо работы",
    },
    {
        "subtype": "inner_child",
        "name": "Внутренний ребёнок",
        "key": "part:inner_child",
        "patterns": [r"хочу чтобы кто-то понял", r"одиноч", r"беспомощ", r"мне страшно"],
        "voice": "Пожалуйста, пойми и не отвергай меня.",
        "default_text": "Мне одиноко и страшно",
    },
]


def _extract_fragment(text: str, pattern: str, default_text: str) -> str:
    match = re.search(pattern, text, flags=re.IGNORECASE)
    if not match:
        return default_text
    start = max(0, match.start() - 20)
    end = min(len(text), match.end() + 30)
    return text[start:end].strip(" .,!?:;")


async def extract(user_id: str, text: str, intent: str, person_id: str) -> tuple[list[Node], list[Edge]]:
    nodes: list[Node] = []
    edges: list[Edge] = []

    lowered = text.lower()
    for rule in PART_RULES:
        matched_pattern = None
        for pattern in rule["patterns"]:
            if re.search(pattern, lowered, flags=re.IGNORECASE):
                matched_pattern = pattern
                break

        if not matched_pattern:
            continue

        part = Node(
            user_id=user_id,
            type="PART",
            subtype=rule["subtype"],
            name=rule["name"],
            key=rule["key"],
            text=_extract_fragment(text, matched_pattern, rule["default_text"]),
            metadata={"voice": rule["voice"]},
        )
        nodes.append(part)
        edges.append(
            Edge(
                user_id=user_id,
                source_node_id=person_id,
                target_node_id=part.id,
                relation="HAS_PART",
            )
        )

    if len(nodes) > 1:
        by_subtype = {node.subtype: node for node in nodes}
        critic = by_subtype.get("critic")
        firefighter = by_subtype.get("firefighter")
        if critic and firefighter:
            edges.append(
                Edge(
                    user_id=user_id,
                    source_node_id=critic.id,
                    target_node_id=firefighter.id,
                    relation="CONFLICTS_WITH",
                )
            )

    return nodes, edges
