from __future__ import annotations

import re

from core.graph.model import Edge, Node


def extract(user_id: str, text: str, intent: str, person_id: str) -> tuple[list[Node], list[Edge]]:
    nodes: list[Node] = []
    edges: list[Edge] = []

    lowered = text.lower()
    if not re.search(r"\b(боюсь|страшно|тревож|рад|злюсь|груст|чувствую)\b", lowered):
        return nodes, edges

    label = "neutral"
    valence = 0.0
    arousal = 0.0

    if re.search(r"\b(боюсь|страшно|тревож)\b", lowered):
        label = "fear"
        valence = -0.6
        arousal = 0.7
    elif re.search(r"\b(груст|печал)\b", lowered):
        label = "sadness"
        valence = -0.7
        arousal = -0.2
    elif re.search(r"\b(рад|счастлив)\b", lowered):
        label = "joy"
        valence = 0.8
        arousal = 0.4

    emotion = Node(
        user_id=user_id,
        type="EMOTION",
        key=None,
        metadata={"valence": valence, "arousal": arousal, "label": label},
    )
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
        edges.append(
            Edge(
                user_id=user_id,
                source_node_id=emotion.id,
                target_node_id=soma.id,
                relation="EXPRESSED_AS",
            )
        )

    return nodes, edges
