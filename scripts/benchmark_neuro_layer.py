from __future__ import annotations

import json

from core.analytics.extraction_quality import Hypothesis, choose_best_hypothesis
from core.graph.model import Node


def _node(node_type: str, *, label: str | None = None, confidence: float = 0.8, valence: float = 0.0, arousal: float = 0.0, name: str | None = None, text: str | None = None, metadata: dict | None = None) -> Node:
    payload = dict(metadata or {})
    if node_type == "EMOTION" and label:
        payload.setdefault("label", label)
        payload.setdefault("confidence", confidence)
        payload.setdefault("valence", valence)
        payload.setdefault("arousal", arousal)
    return Node(user_id="benchmark", type=node_type, name=name, text=text, metadata=payload)


def _clone_hypothesis(hyp: Hypothesis) -> Hypothesis:
    cloned_nodes: list[Node] = []
    for node in hyp.nodes:
        cloned_nodes.append(
            Node(
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
        )
    return Hypothesis(
        name=hyp.name,
        nodes=cloned_nodes,
        edges=list(hyp.edges),
        confidence=hyp.confidence,
        rationale=hyp.rationale,
    )


def build_cases() -> list[tuple[str, list[Hypothesis], str]]:
    cases: list[tuple[str, list[Hypothesis], str]] = []

    cases.append(
        (
            "После встречи с начальником меня трясет и потеют ладони.",
            [
                Hypothesis(
                    name="shame-heavy",
                    nodes=[
                        _node("EVENT", name="встреча"),
                        _node("THOUGHT", text="я плохой"),
                        _node("EMOTION", label="стыд", confidence=0.86, valence=-0.5, arousal=0.2),
                        _node("NEED", name="принятие"),
                        _node("PART", name="Критик"),
                    ],
                    edges=[],
                    confidence=0.86,
                ),
                Hypothesis(
                    name="anxiety-intero",
                    nodes=[
                        _node("EVENT", name="встреча"),
                        _node("EMOTION", label="тревога", confidence=0.8, valence=-0.8, arousal=0.85),
                        _node("SOMA", metadata={"location": "ладони", "sensation": "потеют"}),
                        _node("NEED", name="безопасность"),
                    ],
                    edges=[],
                    confidence=0.8,
                ),
            ],
            "тревога",
        )
    )

    cases.append(
        (
            "Уже неделю почти не сплю, устал и не могу собраться.",
            [
                Hypothesis(
                    name="shame-rationalized",
                    nodes=[
                        _node("THOUGHT", text="я ленивый"),
                        _node("EMOTION", label="стыд", confidence=0.85, valence=-0.55, arousal=0.3),
                        _node("NEED", name="принятие"),
                        _node("PART", name="Критик"),
                    ],
                    edges=[],
                    confidence=0.85,
                ),
                Hypothesis(
                    name="allostatic-fatigue",
                    nodes=[
                        _node("EMOTION", label="усталость", confidence=0.76, valence=-0.45, arousal=-0.35),
                        _node("SOMA", metadata={"location": "тело", "sensation": "изнеможение"}),
                        _node("NEED", name="восстановление"),
                    ],
                    edges=[],
                    confidence=0.76,
                ),
            ],
            "усталость",
        )
    )

    cases.append(
        (
            "После похвалы почувствовал легкость и спокойствие.",
            [
                Hypothesis(
                    name="anxiety-default",
                    nodes=[
                        _node("EMOTION", label="тревога", confidence=0.81, valence=-0.5, arousal=0.65),
                        _node("NEED", name="безопасность"),
                        _node("PART", name="Критик"),
                    ],
                    edges=[],
                    confidence=0.81,
                ),
                Hypothesis(
                    name="relief-social-reward",
                    nodes=[
                        _node("EMOTION", label="облегчение", confidence=0.72, valence=0.55, arousal=0.15),
                        _node("NEED", name="признание"),
                    ],
                    edges=[],
                    confidence=0.72,
                ),
            ],
            "облегчение",
        )
    )

    cases.append(
        (
            "Когда думаю о выступлении перед людьми, сердце стучит и дыхание сбивается.",
            [
                Hypothesis(
                    name="sadness",
                    nodes=[
                        _node("EMOTION", label="грусть", confidence=0.83, valence=-0.5, arousal=-0.2),
                        _node("NEED", name="поддержка"),
                    ],
                    edges=[],
                    confidence=0.83,
                ),
                Hypothesis(
                    name="fear-intero",
                    nodes=[
                        _node("EVENT", name="выступление"),
                        _node("EMOTION", label="страх", confidence=0.78, valence=-0.8, arousal=0.9),
                        _node("SOMA", metadata={"location": "сердце", "sensation": "сильно стучит"}),
                        _node("NEED", name="безопасность"),
                    ],
                    edges=[],
                    confidence=0.78,
                ),
            ],
            "страх",
        )
    )

    return cases


def evaluate(use_neuro_layer: bool) -> dict:
    cases = build_cases()
    snapshots = [
        {"valence_avg": -0.45, "arousal_avg": 0.75},
        {"valence_avg": -0.40, "arousal_avg": 0.65},
        {"valence_avg": -0.35, "arousal_avg": 0.70},
    ]

    hits = 0
    rows = []
    for text, hypotheses, expected in cases:
        selected = choose_best_hypothesis(
            [_clone_hypothesis(hypothesis) for hypothesis in hypotheses],
            recurring_emotions=[],
            text=text,
            recent_snapshots=snapshots,
            use_neuro_layer=use_neuro_layer,
        )
        predicted = ""
        for node in selected.nodes:
            if node.type == "EMOTION":
                predicted = str(node.metadata.get("label", "")).strip().lower()
                break
        ok = predicted == expected
        hits += 1 if ok else 0
        rows.append(
            {
                "text": text,
                "expected": expected,
                "predicted": predicted,
                "selected_hypothesis": selected.name,
                "score": round(selected.score, 4),
                "ok": ok,
            }
        )

    accuracy = hits / len(cases)
    return {
        "samples": len(cases),
        "accuracy": round(accuracy, 4),
        "rows": rows,
    }


def main() -> None:
    baseline = evaluate(use_neuro_layer=False)
    neuro = evaluate(use_neuro_layer=True)
    delta = round(neuro["accuracy"] - baseline["accuracy"], 4)

    result = {
        "baseline": baseline,
        "neuro_layer": neuro,
        "delta_accuracy": delta,
        "improved": delta > 0,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
