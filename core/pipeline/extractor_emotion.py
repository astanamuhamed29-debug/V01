from __future__ import annotations

import re

from core.graph.model import Edge, Node


EMOTION_RULES: list[tuple[re.Pattern[str], str, float, float]] = [
    (re.compile(r"\b(боюсь|страшно|страх|тревож)\w*\b"), "страх", -0.8, 0.6),
    (re.compile(r"\b(стыд|стыдно|стыдом)\w*\b"), "стыд", -0.7, -0.2),
    (re.compile(r"\b(устал|усталость|измотан)\w*\b"), "усталость", -0.5, -0.4),
    (re.compile(r"\b(злость|злюсь|злой)\w*\b"), "злость", -0.7, 0.4),
    (re.compile(r"\b(вина|виноват|виновата)\w*\b"), "вина", -0.6, -0.1),
    (re.compile(r"\b(обида|обидно|обижен|обижена)\w*\b"), "обида", -0.6, -0.2),
    (re.compile(r"\b(груст|печал)\w*\b"), "грусть", -0.7, -0.2),
    (re.compile(r"\b(радость|рад|счастлив)\w*\b"), "радость", 0.8, 0.4),
    (re.compile(r"\b(ступор)\w*\b"), "ступор", -0.4, -0.3),
]


def _emotion_from_word(word: str) -> tuple[str, float, float] | None:
    probe = word.strip().lower()
    for pattern, label, valence, arousal in EMOTION_RULES:
        if pattern.search(probe):
            return label, valence, arousal
    return None


def _detect_emotions(lowered: str) -> list[tuple[str, float, float]]:
    detected: list[tuple[str, float, float]] = []
    seen: set[str] = set()

    between_match = re.search(
        r"(?:что-то\s+)?между\s+([а-яё-]+)\s+и\s+([а-яё-]+)",
        lowered,
        flags=re.IGNORECASE,
    )
    if between_match:
        for token in (between_match.group(1), between_match.group(2)):
            emotion = _emotion_from_word(token)
            if emotion and emotion[0] not in seen:
                seen.add(emotion[0])
                detected.append(emotion)

    for pattern, label, valence, arousal in EMOTION_RULES:
        if pattern.search(lowered) and label not in seen:
            seen.add(label)
            detected.append((label, valence, arousal))

    return detected


async def extract(user_id: str, text: str, intent: str, person_id: str) -> tuple[list[Node], list[Edge]]:
    nodes: list[Node] = []
    edges: list[Edge] = []

    lowered = text.lower()
    if not re.search(r"\b(боюсь|страшно|страх|тревож|рад|радость|злюсь|злость|груст|печал|стыд|устал|вина|обида|ступор|чувствую)\b", lowered):
        return nodes, edges

    emotions = _detect_emotions(lowered)
    if not emotions:
        emotions = [("neutral", 0.0, 0.0)]

    emotion_nodes: list[Node] = []
    for label, valence, arousal in emotions:
        emotion = Node(
            user_id=user_id,
            type="EMOTION",
            key=None,
            metadata={"valence": valence, "arousal": arousal, "label": label},
        )
        emotion_nodes.append(emotion)
        nodes.append(emotion)
        edges.append(
            Edge(
                user_id=user_id,
                source_node_id=person_id,
                target_node_id=emotion.id,
                relation="FEELS",
            )
        )

    body_match = re.search(r"\b(в груди|в животе|в горле|в плечах|в шее)\b", lowered)
    if body_match:
        location = body_match.group(1)
        soma = Node(
            user_id=user_id,
            type="SOMA",
            key=None,
            metadata={"location": location, "sensation": "tension"},
        )
        nodes.append(soma)
        if emotion_nodes:
            edges.append(
                Edge(
                    user_id=user_id,
                    source_node_id=emotion_nodes[0].id,
                    target_node_id=soma.id,
                    relation="EXPRESSED_AS",
                )
            )

    return nodes, edges
