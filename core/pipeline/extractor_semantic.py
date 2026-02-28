from __future__ import annotations

import re

from core.graph.api import normalize_key
from core.graph.model import Edge, Node


PROJECT_DEFINITIONS = [
    ("SELF-OS", "project:self-os", [r"self[- ]?os", r"личн(ая|ую|ой)\s+ос"]),
    ("переезд", "project:переезд", [r"\bпереезд\b", r"\bпереехать\b", r"\bпереезжа\w*\b"]),
]


def _extract_task_text(text: str) -> str:
    match = re.search(r"чтобы\s+(.+)$", text, flags=re.IGNORECASE)
    if match:
        return match.group(1).strip(" .,!?:;")

    match = re.search(r"\bхочу\s+(.+)$", text, flags=re.IGNORECASE)
    if match:
        return match.group(1).strip(" .,!?:;")

    cleaned = re.sub(r"^(надо|нужно|хочу|сделать)\s+", "", text.strip(), flags=re.IGNORECASE)
    return cleaned.strip(" .,!?:;")


def _extract_belief_text(text: str) -> str:
    lowered = text.lower().strip()
    if lowered.startswith("я боюсь"):
        return text.strip().rstrip(".!")
    if lowered.startswith("мне кажется"):
        return text.strip().rstrip(".!")
    return text.strip().rstrip(".!")


def _detect_project(text: str) -> tuple[str, str] | None:
    for name, key, patterns in PROJECT_DEFINITIONS:
        if any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in patterns):
            return name, key
    return None


def _detect_value(lowered: str) -> tuple[str, str] | None:
    if re.search(r"в\s+ч[её]м(?:\s+\w+){0,3}\s+польза", lowered, flags=re.IGNORECASE):
        return "польза", "в чем твоя польза"
    if re.search(r"\b(зачем|какой\s+смысл|в\s+ч[её]м(?:\s+\w+){0,3}\s+смысл|что\s+это\s+(?:мне\s+)?да[её]т)\b", lowered, flags=re.IGNORECASE):
        return "смысл", "какой в этом смысл"
    if re.search(r"\b(более\s+жив\w*|живым|важно\s+чтобы\s+было\s+жив\w*)\b", lowered, flags=re.IGNORECASE):
        return "аутентичность", "хочу сделать вывод более живым"
    return None


async def extract(user_id: str, text: str, intent: str, person_id: str) -> tuple[list[Node], list[Edge]]:
    nodes: list[Node] = []
    edges: list[Edge] = []

    note = Node(
        user_id=user_id,
        type="NOTE",
        text=text,
        key=None,
    )
    nodes.append(note)

    project_node: Node | None = None
    detected_project = _detect_project(text)
    if detected_project:
        project_name, project_key = detected_project
        project_node = Node(
            user_id=user_id,
            type="PROJECT",
            name=project_name,
            key=project_key,
        )
        nodes.append(project_node)
        edges.append(
            Edge(
                user_id=user_id,
                source_node_id=person_id,
                target_node_id=project_node.id,
                relation="OWNS_PROJECT",
            )
        )
        edges.append(
            Edge(
                user_id=user_id,
                source_node_id=note.id,
                target_node_id=project_node.id,
                relation="RELATES_TO",
            )
        )

    if intent == "TASK_LIKE" or re.search(r"\b(надо|нужно|сделать|хочу)\b", text, flags=re.IGNORECASE):
        task_text = _extract_task_text(text)
        task = Node(
            user_id=user_id,
            type="TASK",
            text=task_text,
            key=f"task:{normalize_key(task_text)}",
        )
        nodes.append(task)
        if project_node:
            edges.append(
                Edge(
                    user_id=user_id,
                    source_node_id=project_node.id,
                    target_node_id=task.id,
                    relation="HAS_TASK",
                )
            )

    if re.search(r"\b(я боюсь|мне кажется,? что я|я не вывезу|я не смогу)\b", text, flags=re.IGNORECASE):
        belief_text = _extract_belief_text(text)
        belief = Node(
            user_id=user_id,
            type="BELIEF",
            text=belief_text,
            key=f"belief:{normalize_key(belief_text)}",
        )
        nodes.append(belief)
        edges.append(
            Edge(
                user_id=user_id,
                source_node_id=person_id,
                target_node_id=belief.id,
                relation="HOLDS_BELIEF",
            )
        )
        edges.append(
            Edge(
                user_id=user_id,
                source_node_id=note.id,
                target_node_id=belief.id,
                relation="RELATES_TO",
            )
        )

    lowered = text.lower()
    value_detected = _detect_value(lowered)
    if value_detected:
        value_name, value_text = value_detected
        value = Node(
            user_id=user_id,
            type="VALUE",
            name=value_name,
            text=value_text,
            key=f"value:{normalize_key(value_name)}",
        )
        nodes.append(value)
        edges.append(
            Edge(
                user_id=user_id,
                source_node_id=person_id,
                target_node_id=value.id,
                relation="HAS_VALUE",
            )
        )
        edges.append(
            Edge(
                user_id=user_id,
                source_node_id=note.id,
                target_node_id=value.id,
                relation="RELATES_TO",
            )
        )

    return nodes, edges
