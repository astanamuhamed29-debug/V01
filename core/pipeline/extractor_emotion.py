"""Emotion extraction pipeline: regex → model → LLM arbiter.

Architecture (3-layer hybrid):

Layer 1 — **Regex rules** (< 1 ms, always available)
    Fast keyword matching returning pre-calibrated VAD + intensity.
    Used as the baseline and as a fallback when deeper layers are
    unavailable or filtered out by confidence.

Layer 2 — **Model-based VAD regression + multi-label classification**
    Embedding-backed cosine-similarity to emotion-label centroids.
    Runs when an ``EmbeddingService`` is available.

Layer 3 — **LLM arbiter**
    Invoked **only** when:
      (a) confidence from layers 1-2 is below ``LLM_ARBITER_THRESHOLD``, or
      (b) sarcasm/irony is suspected, or
      (c) VAD from regex conflicts with model-based multi-label.
    Returns structured JSON with labels, VAD, confidence, cause,
    sarcasm flag.

Additional capabilities:
- **ERC context window**: last *N* user messages feed into the
  extraction so that conversational context shapes the result.
- **Personal baseline**: per-user neutral VAD point; all reported
  VAD values are expressed as deltas from baseline.
- Returns **confidence** ∈ [0, 1] for every emotion signal.
- Detects **implicit** emotions (no explicit keyword but context
  implies affect).
- Emits ``cause`` field when cause-phrase is found.

# NOTE: improved architecture — 3-layer hybrid with ERC context,
# personal baselines, and confidence-weighted signals replacing
# the old flat regex table.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from core.graph.model import Edge, Node

if TYPE_CHECKING:
    from core.context.session_memory import SessionMemory
    from core.llm_client import LLMClient

logger = logging.getLogger(__name__)


# ── Tuneable constants ────────────────────────────────────────────
EMOTION_CONFIDENCE_MIN: float = 0.3
LLM_ARBITER_THRESHOLD: float = 0.5
ERC_CONTEXT_WINDOW: int = 5
BASELINE_V: float = 0.0
BASELINE_A: float = 0.0
BASELINE_D: float = 0.0
IMPLICIT_MIN_CONTEXT: int = 2


# ═══════════════════════════════════════════════════════════════════
# Data-transfer objects
# ═══════════════════════════════════════════════════════════════════

@dataclass(slots=True)
class EmotionSignal:
    """Single emotion detected in a message."""

    label: str
    valence: float
    arousal: float
    dominance: float
    intensity: float
    confidence: float = 0.9
    source: str = "regex"        # "regex" | "model" | "llm"
    implicit: bool = False
    sarcasm: bool = False
    cause: str | None = None
    multi_labels: list[str] = field(default_factory=list)

    def to_metadata(self) -> dict[str, Any]:
        meta: dict[str, Any] = {
            "label": self.label,
            "valence": round(self.valence, 3),
            "arousal": round(self.arousal, 3),
            "dominance": round(self.dominance, 3),
            "intensity": round(self.intensity, 3),
            "confidence": round(self.confidence, 3),
            "source": self.source,
            "implicit": self.implicit,
            "sarcasm": self.sarcasm,
        }
        if self.cause:
            meta["cause"] = self.cause
        if self.multi_labels:
            meta["multi_labels"] = self.multi_labels
        return meta


# ═══════════════════════════════════════════════════════════════════
# Layer 1 — Regex rules (fast baseline)
# ═══════════════════════════════════════════════════════════════════

EMOTION_RULES: list[tuple[re.Pattern[str], str, float, float, float, float]] = [
    (re.compile(r"\b(боюсь|страшно|страх|тревож)\w*\b"), "страх", -0.8, 0.6, -0.6, 0.9),
    (re.compile(r"\b(стыд|стыдно|стыдом)\w*\b"), "стыд", -0.7, -0.2, -0.5, 0.8),
    (re.compile(r"\b(устал|усталость|измотан)\w*\b"), "усталость", -0.5, -0.4, -0.3, 0.7),
    (re.compile(r"\b(злость|злюсь|злой|бешен|раздраж)\w*\b"), "злость", -0.7, 0.4, 0.7, 0.85),
    (re.compile(r"\b(вина|виноват|виновата)\w*\b"), "вина", -0.6, -0.1, -0.4, 0.75),
    (re.compile(r"\b(обид|обида|обидно|обижен|обижена)\w*\b"), "обида", -0.6, -0.2, -0.2, 0.7),
    (re.compile(r"\b(груст|печал|тоскл|уныл)\w*\b"), "грусть", -0.7, -0.2, -0.4, 0.7),
    (re.compile(r"\b(радость|рад|счастлив|доволен|довольна)\w*\b"), "радость", 0.8, 0.4, 0.4, 0.8),
    (re.compile(r"\b(ступор|замер|оцепене)\w*\b"), "ступор", -0.4, -0.3, -0.5, 0.65),
    (re.compile(r"\b(отвращен|противно|тошнит)\w*\b"), "отвращение", -0.6, 0.2, 0.3, 0.75),
    (re.compile(r"\b(надежд|верю|оптимизм)\w*\b"), "надежда", 0.5, 0.2, 0.3, 0.6),
    (re.compile(r"\b(одинок|одиночеств)\w*\b"), "одиночество", -0.7, -0.3, -0.5, 0.8),
    # NOTE: improved — added отвращение, надежда, одиночество categories.
    (re.compile(r"ненавижу\s+себя|презираю\s+себя|я\s+никчем"), "стыд", -0.8, -0.3, -0.6, 0.9),
]

# Cause-phrase patterns  # NOTE: improved — cause extraction via regex
_CAUSE_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"(?:из-за|потому что|от того что|когда)\s+(.{3,60}?)(?:[.,!?;]|$)", re.IGNORECASE),
    re.compile(r"(?:после|при|во время)\s+(.{3,40}?)(?:[.,!?;]|$)", re.IGNORECASE),
]

# Sarcasm/irony heuristics  # NOTE: improved — sarcasm detection
_SARCASM_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"(?:ага,? конечно|ну да,? ну да|как же|отличн\w+,? просто)", re.IGNORECASE),
    re.compile(r"(?:😂|🙃|😏|👏)\s*(?:отличн|прекрасн|замечательн|чудесн)", re.IGNORECASE),
]

# GoEmotions-compatible label mapping.
# NOTE: improved — multi-label bridge to GoEmotions taxonomy.
_LABEL_TO_GOEMOTION: dict[str, list[str]] = {
    "страх": ["fear", "nervousness"],
    "стыд": ["embarrassment", "remorse"],
    "усталость": ["annoyance", "disappointment"],
    "злость": ["anger", "annoyance"],
    "вина": ["remorse"],
    "обида": ["disappointment", "sadness"],
    "грусть": ["sadness", "grief"],
    "радость": ["joy", "amusement"],
    "ступор": ["confusion", "nervousness"],
    "отвращение": ["disgust"],
    "надежда": ["optimism"],
    "одиночество": ["sadness", "disappointment"],
}


def _emotion_from_word(word: str) -> tuple[str, float, float, float, float] | None:
    """Match a single word against EMOTION_RULES. Public for tests."""
    probe = word.strip().lower()
    for pattern, label, valence, arousal, dominance, intensity in EMOTION_RULES:
        if pattern.search(probe):
            return label, valence, arousal, dominance, intensity
    return None


def _extract_cause(text: str) -> str | None:
    for pat in _CAUSE_PATTERNS:
        m = pat.search(text)
        if m:
            return m.group(1).strip()
    return None


def _detect_sarcasm(text: str) -> bool:
    return any(pat.search(text) for pat in _SARCASM_PATTERNS)


def _detect_emotions(lowered: str) -> list[EmotionSignal]:
    """Layer 1: fast regex detection returning EmotionSignal objects.

    Backward-compatible: still exposed as ``_detect_emotions`` but now
    returns rich ``EmotionSignal`` instances instead of bare tuples.
    """
    detected: list[EmotionSignal] = []
    seen: set[str] = set()

    between_match = re.search(
        r"(?:что-то\s+)?между\s+([а-яё-]+)\s+и\s+([а-яё-]+)",
        lowered,
        flags=re.IGNORECASE,
    )
    if between_match:
        for token in (between_match.group(1), between_match.group(2)):
            emo = _emotion_from_word(token)
            if emo and emo[0] not in seen:
                seen.add(emo[0])
                detected.append(EmotionSignal(
                    label=emo[0], valence=emo[1], arousal=emo[2],
                    dominance=emo[3], intensity=emo[4],
                    confidence=0.85, source="regex",
                    multi_labels=_LABEL_TO_GOEMOTION.get(emo[0], []),
                ))

    for pattern, label, v, a, d, i in EMOTION_RULES:
        if pattern.search(lowered) and label not in seen:
            seen.add(label)
            detected.append(EmotionSignal(
                label=label, valence=v, arousal=a,
                dominance=d, intensity=i,
                confidence=0.85, source="regex",
                multi_labels=_LABEL_TO_GOEMOTION.get(label, []),
            ))

    if detected:
        cause = _extract_cause(lowered)
        sarcasm = _detect_sarcasm(lowered)
        for sig in detected:
            sig.cause = cause
            sig.sarcasm = sarcasm

    return detected


# ═══════════════════════════════════════════════════════════════════
# Layer 2 — Model-based VAD regression (centroid projection)
# ═══════════════════════════════════════════════════════════════════
# NOTE: improved — pluggable model layer. Current impl uses cosine
# distance to emotion-label centroids as a lightweight proxy for a
# BERT-VAD regressor.

_CENTROID_TEXTS: dict[str, str] = {
    "страх": "мне страшно, я боюсь",
    "стыд": "мне стыдно за себя",
    "злость": "меня бесит, я злюсь",
    "грусть": "мне грустно и печально",
    "радость": "я рад и счастлив",
    "вина": "я виноват, мне стыдно",
    "усталость": "я устал, нет сил",
    "обида": "мне обидно",
    "ступор": "я в ступоре, не могу думать",
    "отвращение": "мне противно",
    "надежда": "я верю, всё получится",
    "одиночество": "я одинок, никого рядом",
}


async def _model_predict(
    text: str,
    embedding_service: Any | None,
) -> list[EmotionSignal]:
    """Layer 2: embedding-based emotion prediction."""
    if embedding_service is None:
        return []

    try:
        text_emb = await embedding_service.embed_text(text)
        if text_emb is None:
            return []
    except Exception as exc:
        logger.debug("Model layer embedding failed: %s", exc)
        return []

    from core.utils.math import cosine_similarity

    results: list[EmotionSignal] = []
    for label, centroid_text in _CENTROID_TEXTS.items():
        try:
            centroid_emb = await embedding_service.embed_text(centroid_text)
            if centroid_emb is None:
                continue
        except Exception:
            continue

        sim = cosine_similarity(text_emb, centroid_emb)
        if sim < 0.45:
            continue

        confidence = min(0.3 + (sim - 0.45) * (0.65 / 0.35), 0.99)

        rule_hit = None
        for _, rl, rv, ra, rd, ri in EMOTION_RULES:
            if rl == label:
                rule_hit = (rv, ra, rd, ri)
                break

        if rule_hit:
            v, a, d, i = rule_hit
        else:
            v, a, d, i = 0.0, 0.0, 0.0, 0.5

        results.append(EmotionSignal(
            label=label,
            valence=v, arousal=a, dominance=d, intensity=i,
            confidence=round(confidence, 3),
            source="model",
            multi_labels=_LABEL_TO_GOEMOTION.get(label, []),
        ))

    results.sort(key=lambda s: s.confidence, reverse=True)
    return results[:3]


# ═══════════════════════════════════════════════════════════════════
# Layer 3 — LLM arbiter
# ═══════════════════════════════════════════════════════════════════

_LLM_EMOTION_PROMPT = """\
Ты — специалист по распознаванию эмоций.  Проанализируй текст и верни
JSON (и ТОЛЬКО JSON, без markdown) со следующей структурой:
{
  "emotions": [
    {
      "label": "<метка эмоции на русском>",
      "valence": <float -1..1>,
      "arousal": <float -1..1>,
      "dominance": <float -1..1>,
      "intensity": <float 0..1>,
      "confidence": <float 0..1>,
      "cause": "<причина или null>",
      "sarcasm": <bool>,
      "implicit": <bool>
    }
  ]
}
Правила:
- Верни от 0 до 3 эмоций
- implicit=true если эмоция не названа прямо, а следует из контекста
- Учитывай предыдущие реплики (контекст) при оценке
- Если текст нейтральный — верни пустой массив emotions
"""


async def _llm_arbitrate(
    text: str,
    context_window: list[str],
    llm_client: "LLMClient | None",
) -> list[EmotionSignal]:
    """Layer 3: invoke LLM for ambiguous/low-confidence cases."""
    if llm_client is None:
        return []

    context_block = "\n".join(f"[контекст] {msg}" for msg in context_window[-ERC_CONTEXT_WINDOW:])
    user_payload = f"{context_block}\n\n[текущее сообщение] {text}" if context_block else text

    try:
        raw = await llm_client.extract_emotion(user_payload, "FEELING_REPORT")
    except Exception as exc:
        logger.warning("LLM emotion arbiter failed: %s", exc)
        return []

    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError:
            return []

    if not isinstance(raw, dict):
        return []

    emotions_raw = raw.get("emotions", [])
    if not isinstance(emotions_raw, list):
        return []

    signals: list[EmotionSignal] = []
    for item in emotions_raw[:3]:
        if not isinstance(item, dict) or "label" not in item:
            continue
        signals.append(EmotionSignal(
            label=str(item["label"]),
            valence=float(item.get("valence", 0)),
            arousal=float(item.get("arousal", 0)),
            dominance=float(item.get("dominance", 0)),
            intensity=float(item.get("intensity", 0.5)),
            confidence=float(item.get("confidence", 0.7)),
            source="llm",
            implicit=bool(item.get("implicit", False)),
            sarcasm=bool(item.get("sarcasm", False)),
            cause=item.get("cause"),
            multi_labels=_LABEL_TO_GOEMOTION.get(str(item["label"]), []),
        ))

    return signals


# ═══════════════════════════════════════════════════════════════════
# ERC context window  # NOTE: improved — conversational context
# ═══════════════════════════════════════════════════════════════════

def _build_context_window(
    session_memory: "SessionMemory | None",
    user_id: str,
) -> list[str]:
    if session_memory is None:
        return []
    ctx = session_memory.get_context(user_id, max_messages=ERC_CONTEXT_WINDOW * 2)
    return [m["text"] for m in ctx if m.get("role") == "user"]


# ═══════════════════════════════════════════════════════════════════
# Personal baseline  # NOTE: improved — per-user neutral VAD point
# ═══════════════════════════════════════════════════════════════════

@dataclass(slots=True)
class PersonalBaseline:
    """Per-user neutral VAD reference point (EMA)."""

    valence: float = BASELINE_V
    arousal: float = BASELINE_A
    dominance: float = BASELINE_D
    sample_count: int = 0

    def update(self, v: float, a: float, d: float, alpha: float = 0.05) -> None:
        if self.sample_count == 0:
            self.valence = v
            self.arousal = a
            self.dominance = d
        else:
            self.valence += alpha * (v - self.valence)
            self.arousal += alpha * (a - self.arousal)
            self.dominance += alpha * (d - self.dominance)
        self.sample_count += 1

    def delta(self, v: float, a: float, d: float) -> tuple[float, float, float]:
        return (v - self.valence, a - self.arousal, d - self.dominance)

    def to_dict(self) -> dict[str, Any]:
        return {
            "baseline_v": round(self.valence, 3),
            "baseline_a": round(self.arousal, 3),
            "baseline_d": round(self.dominance, 3),
            "baseline_samples": self.sample_count,
        }


_baselines: dict[str, PersonalBaseline] = {}


def get_baseline(user_id: str) -> PersonalBaseline:
    if user_id not in _baselines:
        _baselines[user_id] = PersonalBaseline()
    return _baselines[user_id]


def load_baseline_from_meta(user_id: str, meta: dict[str, Any]) -> None:
    bl = PersonalBaseline(
        valence=float(meta.get("baseline_v", BASELINE_V)),
        arousal=float(meta.get("baseline_a", BASELINE_A)),
        dominance=float(meta.get("baseline_d", BASELINE_D)),
        sample_count=int(meta.get("baseline_samples", 0)),
    )
    _baselines[user_id] = bl


# ═══════════════════════════════════════════════════════════════════
# Fusion / merge logic
# ═══════════════════════════════════════════════════════════════════

def _merge_signals(
    regex_signals: list[EmotionSignal],
    model_signals: list[EmotionSignal],
    llm_signals: list[EmotionSignal],
) -> list[EmotionSignal]:
    """Fuse signals from all layers: prefer higher-confidence source.

    Priority: LLM > model > regex (when same label appears).
    # NOTE: improved — multi-source fusion with confidence ranking.
    """
    by_label: dict[str, EmotionSignal] = {}

    for sig in regex_signals:
        by_label[sig.label] = sig
    for sig in model_signals:
        existing = by_label.get(sig.label)
        if existing is None or sig.confidence > existing.confidence:
            by_label[sig.label] = sig
    for sig in llm_signals:
        existing = by_label.get(sig.label)
        if existing is None or sig.confidence >= existing.confidence:
            by_label[sig.label] = sig

    result = [s for s in by_label.values() if s.confidence >= EMOTION_CONFIDENCE_MIN]
    result.sort(key=lambda s: s.confidence, reverse=True)
    return result


# ═══════════════════════════════════════════════════════════════════
# Guard regex
# ═══════════════════════════════════════════════════════════════════

_EMOTION_GUARD = re.compile(
    r"(боюсь|страшно|страх|тревож|рад|радость|злюсь|злость|бешен|раздраж"
    r"|груст|печал|тоскл|уныл|стыд|устал|вина|обид|ступор|замер|оцепен"
    r"|чувствую|ненавижу\s+себя|презираю\s+себя|отвращен|противно|тошнит"
    r"|надежд|верю|оптимизм|одинок|одиночеств|доволен|довольна|счастлив"
    r"|плохо|хорошо|нормально|ок|паник|нервнич|волну|беспокой)",
    re.IGNORECASE,
)


# ═══════════════════════════════════════════════════════════════════
# Public API
# ═══════════════════════════════════════════════════════════════════

async def extract(
    user_id: str,
    text: str,
    intent: str,
    person_id: str,
    *,
    session_memory: "SessionMemory | None" = None,
    llm_client: "LLMClient | None" = None,
    embedding_service: Any | None = None,
) -> tuple[list[Node], list[Edge]]:
    """Full emotion extraction pipeline.

    Parameters
    ----------
    user_id, text, intent, person_id : str
        Core identification / message fields.
    session_memory : SessionMemory, optional
        For ERC context window.
    llm_client : LLMClient, optional
        For Layer 3 arbitration.
    embedding_service : EmbeddingService, optional
        For Layer 2 model-based prediction.

    Returns
    -------
    tuple[list[Node], list[Edge]]
        Emotion/Soma nodes and FEELS/EXPRESSED_AS edges.
    """
    nodes: list[Node] = []
    edges: list[Edge] = []
    lowered = text.lower()

    # ERC context window  # NOTE: improved — contextual emotion recognition
    context_window = _build_context_window(session_memory, user_id)

    # ── Layer 1: regex ──────────────────────────────────────────
    regex_signals = _detect_emotions(lowered)

    # Guard: skip deep layers if no emotional keywords AND intent
    # is not feeling-related.
    if not regex_signals and not _EMOTION_GUARD.search(lowered):
        if intent not in ("FEELING_REPORT", "REFLECTION"):
            return nodes, edges

    # ── Layer 2: model ──────────────────────────────────────────
    model_signals = await _model_predict(text, embedding_service)

    # ── Layer 3: LLM arbiter (conditional) ──────────────────────
    llm_signals: list[EmotionSignal] = []
    needs_arbiter = (
        (regex_signals and all(s.confidence < LLM_ARBITER_THRESHOLD for s in regex_signals))
        or _detect_sarcasm(lowered)
        or (
            regex_signals
            and model_signals
            and regex_signals[0].label != model_signals[0].label
        )
        or (not regex_signals and intent == "FEELING_REPORT")
        or (
            not regex_signals
            and len(context_window) >= IMPLICIT_MIN_CONTEXT
            and intent in ("FEELING_REPORT", "REFLECTION")
        )
    )
    if needs_arbiter:
        llm_signals = await _llm_arbitrate(text, context_window, llm_client)

    # ── Fusion ──────────────────────────────────────────────────
    merged = _merge_signals(regex_signals, model_signals, llm_signals)

    if not merged:
        return nodes, edges

    # ── Personal baseline update ────────────────────────────────
    baseline = get_baseline(user_id)
    for sig in merged:
        baseline.update(sig.valence, sig.arousal, sig.dominance)

    # ── Build nodes & edges ─────────────────────────────────────
    # NOTE: improved — key is now None (unique UUID per signal)
    # instead of date-based key that collapsed intra-session trajectory.
    now_iso = datetime.now(timezone.utc).isoformat()
    emotion_nodes: list[Node] = []

    for sig in merged:
        dv, da, dd = baseline.delta(sig.valence, sig.arousal, sig.dominance)
        meta = sig.to_metadata()
        meta["delta_v"] = round(dv, 3)
        meta["delta_a"] = round(da, 3)
        meta["delta_d"] = round(dd, 3)
        meta["created_at"] = now_iso

        emotion = Node(
            user_id=user_id,
            type="EMOTION",
            key=None,
            metadata=meta,
        )
        emotion_nodes.append(emotion)
        nodes.append(emotion)
        edges.append(Edge(
            user_id=user_id,
            source_node_id=person_id,
            target_node_id=emotion.id,
            relation="FEELS",
        ))

    # ── Somatic markers ─────────────────────────────────────────
    body_match = re.search(
        r"\b(в груди|в животе|в горле|в плечах|в шее|в голове|в спине)\b",
        lowered,
    )
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
            edges.append(Edge(
                user_id=user_id,
                source_node_id=emotion_nodes[0].id,
                target_node_id=soma.id,
                relation="EXPRESSED_AS",
            ))

    return nodes, edges
