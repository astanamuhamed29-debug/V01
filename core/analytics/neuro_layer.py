from __future__ import annotations

from dataclasses import dataclass, field

from core.defaults import (
    NEURO_ALLOSTATIC_HIGH,
    NEURO_ALLOSTATIC_RECENT_WINDOW,
)
from core.graph.model import Node


@dataclass(slots=True)
class RDoCAxis:
    valence: float
    arousal: float
    control_load: float
    social_threat_reward: float


@dataclass(slots=True)
class AllostaticState:
    sleep_pressure: float
    fatigue: float
    stress_accumulation: float
    latent_load: float


@dataclass(slots=True)
class InteroceptivePriors:
    emotion_priors: dict[str, float] = field(default_factory=dict)


@dataclass(slots=True)
class ActiveInferenceState:
    violated_needs: list[str]
    prediction_error: float
    counter_hypothesis_penalty: float


@dataclass(slots=True)
class NeuroLayerReport:
    rdoc: RDoCAxis
    allostatic: AllostaticState
    interoception: InteroceptivePriors
    active_inference: ActiveInferenceState

    def to_dict(self) -> dict[str, float | list[str] | dict[str, float]]:
        return {
            "rdoc": {
                "valence": round(self.rdoc.valence, 4),
                "arousal": round(self.rdoc.arousal, 4),
                "control_load": round(self.rdoc.control_load, 4),
                "social_threat_reward": round(self.rdoc.social_threat_reward, 4),
            },
            "allostatic": {
                "sleep_pressure": round(self.allostatic.sleep_pressure, 4),
                "fatigue": round(self.allostatic.fatigue, 4),
                "stress_accumulation": round(self.allostatic.stress_accumulation, 4),
                "latent_load": round(self.allostatic.latent_load, 4),
            },
            "interoception": {
                "emotion_priors": {
                    key: round(value, 4)
                    for key, value in self.interoception.emotion_priors.items()
                }
            },
            "active_inference": {
                "violated_needs": list(self.active_inference.violated_needs),
                "prediction_error": round(self.active_inference.prediction_error, 4),
                "counter_hypothesis_penalty": round(self.active_inference.counter_hypothesis_penalty, 4),
            },
        }


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def _extract_labels(nodes: list[Node], node_type: str) -> list[str]:
    labels: list[str] = []
    for node in nodes:
        if node.type != node_type:
            continue
        if node_type == "EMOTION":
            label = str(node.metadata.get("label", "")).strip().lower()
            if label:
                labels.append(label)
            continue
        raw = node.name or node.text or ""
        if raw:
            labels.append(str(raw).strip().lower())
    return labels


def infer_rdoc_axis(nodes: list[Node], text: str) -> RDoCAxis:
    emotions = [node for node in nodes if node.type == "EMOTION"]
    if emotions:
        node_valence = sum(float(node.metadata.get("valence", 0.0)) for node in emotions) / len(emotions)
        node_arousal = sum(float(node.metadata.get("arousal", 0.0)) for node in emotions) / len(emotions)
    else:
        node_valence = 0.0
        node_arousal = 0.0

    lowered = text.lower()
    positive_hits = sum(
        1
        for token in (
            "похвал",
            "поддерж",
            "рад",
            "легк",
            "спокой",
            "облегч",
            "горд",
        )
        if token in lowered
    )
    negative_hits = sum(
        1
        for token in (
            "страш",
            "трев",
            "стыд",
            "вина",
            "не успеваю",
            "паник",
            "устал",
            "критик",
        )
        if token in lowered
    )
    lexical_valence = _clamp01(0.5 + 0.12 * positive_hits - 0.12 * negative_hits) * 2 - 1

    lexical_arousal_hits = sum(
        1
        for token in (
            "тряс",
            "поте",
            "сердце",
            "паник",
            "сбивает",
            "давит",
            "горит",
        )
        if token in lowered
    )
    lexical_arousal = _clamp01(0.15 * lexical_arousal_hits + (0.08 * negative_hits)) * 2 - 1

    valence = (0.55 * lexical_valence) + (0.45 * node_valence)
    arousal = (0.50 * lexical_arousal) + (0.50 * node_arousal)

    load_hits = sum(
        1
        for token in (
            "не успеваю",
            "перегруз",
            "должен",
            "катастроф",
            "panic",
            "паник",
            "стыд",
            "вина",
        )
        if token in lowered
    )
    control_load = _clamp01((load_hits / 5.0) + max(0.0, arousal * 0.35) + max(0.0, -valence * 0.25))

    social_threat_hits = sum(
        1
        for token in ("увол", "осуд", "началь", "критик", "стыд", "не любят", "одобр")
        if token in lowered
    )
    social_reward_hits = sum(
        1 for token in ("поддерж", "принял", "горж", "похвал", "довер") if token in lowered
    )
    social_threat_reward = _clamp01(0.5 + 0.12 * social_reward_hits - 0.15 * social_threat_hits)

    return RDoCAxis(
        valence=valence,
        arousal=arousal,
        control_load=control_load,
        social_threat_reward=social_threat_reward,
    )


def infer_interoceptive_priors(nodes: list[Node], text: str) -> InteroceptivePriors:
    lowered = text.lower()
    soma_text = " ".join(
        [
            f"{str(node.metadata.get('location', ''))} {str(node.metadata.get('sensation', ''))}"
            for node in nodes
            if node.type == "SOMA"
        ]
    ).lower()
    combined = f"{lowered} {soma_text}"

    priors: dict[str, float] = {}

    if any(token in combined for token in ("тряс", "поте", "сердце", "дыш", "ком в горле")):
        priors["тревога"] = max(priors.get("тревога", 0.0), 0.85)
        priors["страх"] = max(priors.get("страх", 0.0), 0.7)

    if any(token in combined for token in ("тяжесть", "опустош", "вял", "слез")):
        priors["грусть"] = max(priors.get("грусть", 0.0), 0.75)

    if any(token in combined for token in ("напряж", "челюст", "сжат", "жар")):
        priors["злость"] = max(priors.get("злость", 0.0), 0.65)

    if any(token in combined for token in ("устал", "не спал", "бессон", "сонлив")):
        priors["усталость"] = max(priors.get("усталость", 0.0), 0.85)

    return InteroceptivePriors(emotion_priors=priors)


def infer_allostatic_state(text: str, recent_snapshots: list[dict] | None = None) -> AllostaticState:
    lowered = text.lower()

    sleep_pressure = 0.0
    if any(token in lowered for token in ("не спал", "бессон", "сонлив", "мало сна")):
        sleep_pressure = 0.9

    fatigue = 0.0
    if any(token in lowered for token in ("устал", "выгор", "нет сил", "измот")):
        fatigue = 0.85

    snapshots = list(recent_snapshots or [])[:NEURO_ALLOSTATIC_RECENT_WINDOW]
    stress_accumulation = 0.0
    if snapshots:
        neg = [s for s in snapshots if float(s.get("valence_avg", 0.0)) < -0.2]
        hi_ar = [s for s in snapshots if float(s.get("arousal_avg", 0.0)) > 0.45]
        stress_accumulation = _clamp01((len(neg) / max(1, len(snapshots))) * 0.6 + (len(hi_ar) / max(1, len(snapshots))) * 0.6)

    latent_load = _clamp01(0.35 * sleep_pressure + 0.30 * fatigue + 0.35 * stress_accumulation)

    return AllostaticState(
        sleep_pressure=sleep_pressure,
        fatigue=fatigue,
        stress_accumulation=stress_accumulation,
        latent_load=latent_load,
    )


def infer_active_inference_state(nodes: list[Node], text: str, rdoc: RDoCAxis) -> ActiveInferenceState:
    needs = _extract_labels(nodes, "NEED")
    emotions = _extract_labels(nodes, "EMOTION")
    lowered = text.lower()

    violated_needs = list(dict.fromkeys(needs))
    if not violated_needs:
        if any(token in lowered for token in ("увол", "опас", "трев", "страх")):
            violated_needs.append("безопасность")
        if any(token in lowered for token in ("стыд", "вина", "не любят", "критик")):
            violated_needs.append("принятие")
        if any(token in lowered for token in ("давят", "должен", "застав", "контроль")):
            violated_needs.append("автономия")

    prediction_error = _clamp01(max(0.0, -rdoc.valence) * 0.55 + rdoc.control_load * 0.45)

    has_need_edges = False
    for node in nodes:
        if node.type == "NEED":
            has_need_edges = True
            break

    counter_hypothesis_penalty = 0.0
    if prediction_error > 0.65 and not has_need_edges:
        counter_hypothesis_penalty += 0.25
    if prediction_error > 0.65 and not violated_needs:
        counter_hypothesis_penalty += 0.20
    if "тревога" in emotions and "безопасность" not in violated_needs:
        counter_hypothesis_penalty += 0.15

    return ActiveInferenceState(
        violated_needs=violated_needs,
        prediction_error=prediction_error,
        counter_hypothesis_penalty=_clamp01(counter_hypothesis_penalty),
    )


def analyze_neuro_layer(
    *,
    nodes: list[Node],
    text: str,
    recent_snapshots: list[dict] | None = None,
) -> NeuroLayerReport:
    rdoc = infer_rdoc_axis(nodes, text)
    intero = infer_interoceptive_priors(nodes, text)
    allostatic = infer_allostatic_state(text, recent_snapshots=recent_snapshots)
    active = infer_active_inference_state(nodes, text, rdoc=rdoc)
    return NeuroLayerReport(
        rdoc=rdoc,
        allostatic=allostatic,
        interoception=intero,
        active_inference=active,
    )


def neuro_hypothesis_bonus(
    *,
    nodes: list[Node],
    text: str,
    recent_snapshots: list[dict] | None = None,
) -> tuple[float, dict[str, float]]:
    report = analyze_neuro_layer(nodes=nodes, text=text, recent_snapshots=recent_snapshots)
    emotions = set(_extract_labels(nodes, "EMOTION"))

    intero_match = 0.0
    if report.interoception.emotion_priors:
        total_prior = sum(report.interoception.emotion_priors.values()) or 1.0
        hit = sum(
            value
            for label, value in report.interoception.emotion_priors.items()
            if label in emotions
        )
        intero_match = _clamp01(hit / total_prior)
    else:
        intero_match = 0.5

    need_coherence = 1.0 if report.active_inference.violated_needs else 0.0

    social_bias = report.rdoc.social_threat_reward
    positive_emotions = {"радость", "облегчение", "спокойствие", "гордость"}
    negative_emotions = {"тревога", "страх", "грусть", "стыд", "злость", "усталость", "беспомощность"}
    emotion_alignment_terms: list[float] = []
    for label in emotions:
        if label in positive_emotions:
            emotion_alignment_terms.append(_clamp01(0.35 + 0.85 * social_bias))
        elif label in negative_emotions:
            emotion_alignment_terms.append(_clamp01(0.35 + 0.85 * (1.0 - social_bias)))
        else:
            emotion_alignment_terms.append(0.5)
    emotion_alignment = sum(emotion_alignment_terms) / len(emotion_alignment_terms) if emotion_alignment_terms else 0.5

    overload = report.allostatic.latent_load
    overload_alignment = 0.5
    if overload >= NEURO_ALLOSTATIC_HIGH:
        if emotions & {"усталость", "грусть", "тревога", "страх"}:
            overload_alignment = 1.0
        elif emotions & {"стыд", "злость"}:
            overload_alignment = 0.45
        else:
            overload_alignment = 0.2
    elif overload <= 0.35:
        overload_alignment = 1.0 if emotions & {"спокойствие", "радость", "облегчение"} else 0.45

    rdoc_alignment = _clamp01(
        (0.5 + (-report.rdoc.valence * 0.25) + (report.rdoc.arousal * 0.25))
        if emotions & {"тревога", "страх", "грусть", "злость", "стыд", "усталость"}
        else (0.5 + (report.rdoc.valence * 0.30) - (report.rdoc.control_load * 0.20))
    )

    intero_mismatch_penalty = 0.0
    if report.interoception.emotion_priors:
        intero_mismatch_penalty = 1.0 - intero_match

    bonus = (
        0.30 * intero_match
        + 0.15 * need_coherence
        + 0.15 * overload_alignment
        + 0.20 * rdoc_alignment
        + 0.20 * emotion_alignment
        - 0.30 * report.active_inference.counter_hypothesis_penalty
        - 0.45 * intero_mismatch_penalty
    )
    bonus = _clamp01(bonus)

    diagnostics = {
        "neuro_bonus": round(bonus, 6),
        "intero_match": round(intero_match, 6),
        "intero_mismatch_penalty": round(intero_mismatch_penalty, 6),
        "need_coherence": round(float(need_coherence), 6),
        "overload_alignment": round(overload_alignment, 6),
        "rdoc_alignment": round(rdoc_alignment, 6),
        "emotion_alignment": round(emotion_alignment, 6),
        "allostatic_load": round(report.allostatic.latent_load, 6),
        "prediction_error": round(report.active_inference.prediction_error, 6),
        "counter_penalty": round(report.active_inference.counter_hypothesis_penalty, 6),
    }
    return bonus, diagnostics
