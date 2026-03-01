from __future__ import annotations

from core.analytics.extraction_quality import Hypothesis, choose_best_hypothesis
from core.graph.model import Edge, Node


def _node(
    node_type: str,
    *,
    name: str | None = None,
    text: str | None = None,
    metadata: dict | None = None,
) -> Node:
    return Node(
        user_id="u-neuro",
        type=node_type,
        name=name,
        text=text,
        metadata=metadata or {},
    )


def _edge(source: Node, target: Node, relation: str) -> Edge:
    return Edge(
        user_id="u-neuro",
        source_node_id=source.id,
        target_node_id=target.id,
        relation=relation,
    )


def _clone_hypothesis(hyp: Hypothesis) -> Hypothesis:
    node_map: dict[str, Node] = {}
    new_nodes: list[Node] = []
    for node in hyp.nodes:
        copied = Node(
            id=node.id,
            user_id=node.user_id,
            type=node.type,
            name=node.name,
            text=node.text,
            subtype=node.subtype,
            key=node.key,
            metadata=dict(node.metadata),
            created_at=node.created_at,
        )
        new_nodes.append(copied)
        node_map[node.id] = copied

    new_edges: list[Edge] = []
    for edge in hyp.edges:
        new_edges.append(
            Edge(
                id=edge.id,
                user_id=edge.user_id,
                source_node_id=edge.source_node_id,
                target_node_id=edge.target_node_id,
                relation=edge.relation,
                metadata=dict(edge.metadata),
                created_at=edge.created_at,
            )
        )

    return Hypothesis(
        name=hyp.name,
        nodes=new_nodes,
        edges=new_edges,
        confidence=hyp.confidence,
        rationale=hyp.rationale,
    )


def test_neuro_layer_prefers_interoceptive_consistent_hypothesis():
    text = "После разговора с начальником меня трясет, руки потеют и сердце колотится."

    event_a = _node("EVENT", name="разговор")
    thought_a = _node("THOUGHT", text="я плохой сотрудник")
    emotion_a = _node("EMOTION", metadata={"label": "стыд", "confidence": 0.88, "valence": -0.5, "arousal": 0.3})
    need_a = _node("NEED", name="принятие")
    part_a = _node("PART", name="Критик")
    hyp_a = Hypothesis(
        name="shame-heavy",
        nodes=[event_a, thought_a, emotion_a, need_a, part_a],
        edges=[
            _edge(event_a, thought_a, "TRIGGERS"),
            _edge(thought_a, emotion_a, "TRIGGERS"),
            _edge(emotion_a, need_a, "SIGNALS_NEED"),
            _edge(part_a, need_a, "PROTECTS_NEED"),
        ],
        confidence=0.88,
    )

    event_b = _node("EVENT", name="разговор")
    emotion_b = _node("EMOTION", metadata={"label": "тревога", "confidence": 0.82, "valence": -0.8, "arousal": 0.8})
    soma_b = _node("SOMA", metadata={"location": "руки", "sensation": "потеют"})
    need_b = _node("NEED", name="безопасность")
    hyp_b = Hypothesis(
        name="anxiety-interoceptive",
        nodes=[event_b, emotion_b, soma_b, need_b],
        edges=[
            _edge(event_b, emotion_b, "TRIGGERS"),
            _edge(emotion_b, need_b, "SIGNALS_NEED"),
            _edge(emotion_b, soma_b, "EXPRESSED_AS"),
        ],
        confidence=0.82,
    )

    recurring: list[dict] = []
    snapshots = [
        {"valence_avg": -0.6, "arousal_avg": 0.7},
        {"valence_avg": -0.5, "arousal_avg": 0.8},
        {"valence_avg": -0.4, "arousal_avg": 0.7},
    ]

    no_neuro = choose_best_hypothesis(
        [_clone_hypothesis(hyp_a), _clone_hypothesis(hyp_b)],
        recurring_emotions=recurring,
        text=text,
        recent_snapshots=snapshots,
        use_neuro_layer=False,
    )
    with_neuro = choose_best_hypothesis(
        [_clone_hypothesis(hyp_a), _clone_hypothesis(hyp_b)],
        recurring_emotions=recurring,
        text=text,
        recent_snapshots=snapshots,
        use_neuro_layer=True,
    )

    assert no_neuro.name == "shame-heavy"
    assert with_neuro.name == "anxiety-interoceptive"


def test_neuro_layer_benchmark_accuracy_improves_on_counter_hypotheses():
    cases: list[tuple[str, list[Hypothesis], str]] = []

    text1 = "После встречи с начальником меня трясет и потеют ладони."
    h1 = Hypothesis(
        name="c1-shame",
        nodes=[
            _node("EVENT", name="встреча"),
            _node("THOUGHT", text="я плохой"),
            _node("EMOTION", metadata={"label": "стыд", "confidence": 0.86, "valence": -0.5, "arousal": 0.2}),
            _node("NEED", name="принятие"),
            _node("PART", name="Критик"),
        ],
        edges=[],
        confidence=0.86,
    )
    h2 = Hypothesis(
        name="c1-anxiety",
        nodes=[
            _node("EVENT", name="встреча"),
            _node("EMOTION", metadata={"label": "тревога", "confidence": 0.8, "valence": -0.8, "arousal": 0.85}),
            _node("SOMA", metadata={"location": "ладони", "sensation": "потеют"}),
            _node("NEED", name="безопасность"),
        ],
        edges=[],
        confidence=0.8,
    )
    cases.append((text1, [h1, h2], "тревога"))

    text2 = "Уже неделю почти не сплю, устал и не могу собраться."
    h3 = Hypothesis(
        name="c2-shame",
        nodes=[
            _node("THOUGHT", text="я ленивый"),
            _node("EMOTION", metadata={"label": "стыд", "confidence": 0.86, "valence": -0.6, "arousal": 0.25}),
            _node("NEED", name="принятие"),
            _node("PART", name="Критик"),
        ],
        edges=[],
        confidence=0.86,
    )
    h4 = Hypothesis(
        name="c2-fatigue",
        nodes=[
            _node("EMOTION", metadata={"label": "усталость", "confidence": 0.78, "valence": -0.45, "arousal": -0.35}),
            _node("SOMA", metadata={"location": "тело", "sensation": "изнеможение"}),
            _node("NEED", name="восстановление"),
        ],
        edges=[],
        confidence=0.78,
    )
    cases.append((text2, [h3, h4], "усталость"))

    text3 = "После похвалы почувствовал легкость и спокойствие."
    h5 = Hypothesis(
        name="c3-anxiety",
        nodes=[
            _node("EMOTION", metadata={"label": "тревога", "confidence": 0.81, "valence": -0.5, "arousal": 0.65}),
            _node("NEED", name="безопасность"),
            _node("PART", name="Критик"),
        ],
        edges=[],
        confidence=0.81,
    )
    h6 = Hypothesis(
        name="c3-relief",
        nodes=[
            _node("EMOTION", metadata={"label": "облегчение", "confidence": 0.72, "valence": 0.5, "arousal": 0.1}),
            _node("NEED", name="признание"),
        ],
        edges=[],
        confidence=0.72,
    )
    cases.append((text3, [h5, h6], "облегчение"))

    snapshots = [
        {"valence_avg": -0.4, "arousal_avg": 0.7},
        {"valence_avg": -0.45, "arousal_avg": 0.75},
        {"valence_avg": -0.35, "arousal_avg": 0.65},
    ]

    def _accuracy(use_neuro_layer: bool) -> float:
        ok = 0
        for text, hyps, expected in cases:
            selected = choose_best_hypothesis(
                [_clone_hypothesis(hyp) for hyp in hyps],
                recurring_emotions=[],
                text=text,
                recent_snapshots=snapshots,
                use_neuro_layer=use_neuro_layer,
            )
            labels = {
                str(node.metadata.get("label", "")).strip().lower()
                for node in selected.nodes
                if node.type == "EMOTION"
            }
            if expected in labels:
                ok += 1
        return ok / len(cases)

    baseline = _accuracy(use_neuro_layer=False)
    improved = _accuracy(use_neuro_layer=True)

    assert improved > baseline
    assert (improved - baseline) >= (1 / len(cases))
