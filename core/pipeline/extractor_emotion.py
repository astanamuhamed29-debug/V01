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
_BASE_CONFIDENCE: float = 0.75  # base for dynamic confidence calculation

# ── Negation / Intensifier / Uncertainty word-sets ────────────────
_NEGATED_DIMINISHERS: frozenset[str] = frozenset({
    "не очень", "не сильно", "не особо", "не так уж", "не слишком",
})
_AMPLIFIER_WORDS: frozenset[str] = frozenset({
    "очень", "сильно", "дико", "ужасно", "невероятно",
    "крайне", "чрезвычайно", "жутко", "адски", "безумно",
})
_DIMINISHER_WORDS: frozenset[str] = frozenset({
    "немного", "слегка", "чуть", "легко", "едва", "еле", "слабо",
})
_NEGATION_WORDS: frozenset[str] = frozenset({
    "не", "нет", "нету", "ни", "без", "никак", "нисколько", "никогда",
})
_UNCERTAINTY_WORDS: frozenset[str] = frozenset({
    "может", "наверное", "кажется", "вроде", "будто",
    "словно", "типа", "походу",
})


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
    ambivalent: bool = False

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
        if self.ambivalent:
            meta["ambivalent"] = True
        return meta


# ═══════════════════════════════════════════════════════════════════
# Research-backed VAD norms (Warriner et al., 2013 — adapted)
# ═══════════════════════════════════════════════════════════════════
# (valence, arousal, dominance, default_intensity)  — scales [-1..1] / [0..1]

_VAD_NORMS: dict[str, tuple[float, float, float, float]] = {
    "страх":        (-0.55,  0.33, -0.39, 0.85),
    "стыд":         (-0.63,  0.01, -0.55, 0.80),
    "усталость":    (-0.48, -0.65, -0.32, 0.70),
    "злость":       (-0.67,  0.54,  0.05, 0.85),
    "вина":         (-0.72,  0.02, -0.43, 0.75),
    "обида":        (-0.70,  0.16, -0.47, 0.70),
    "грусть":       (-0.73, -0.38, -0.39, 0.70),
    "радость":      ( 0.87,  0.37,  0.55, 0.80),
    "ступор":       (-0.16, -0.40, -0.38, 0.65),
    "отвращение":   (-0.64,  0.23,  0.04, 0.75),
    "надежда":      ( 0.63,  0.14,  0.38, 0.60),
    "одиночество":  (-0.71, -0.12, -0.51, 0.80),
}


def _vad(label: str) -> tuple[float, float, float, float]:
    """Look up VAD + intensity from research norms."""
    return _VAD_NORMS.get(label, (0.0, 0.0, 0.0, 0.5))


# ═══════════════════════════════════════════════════════════════════
# Layer 1 — Regex rules (fast baseline)
# ═══════════════════════════════════════════════════════════════════
# Patterns reference labels; VAD values come from ``_VAD_NORMS``.
# Morphology: every stem family is covered (тревож + тревог, etc.)

_LABEL_PATTERNS: list[tuple[str, str]] = [
    ("страх",       r"\b(боюсь|страшно|страх|тревож|тревог|беспокой|паник|нервнич|волну)\w*\b"),
    ("стыд",        r"\b(стыд|стыдно|стыдом)\w*\b"),
    ("усталость",   r"\b(устал|усталость|измотан|вымотан)\w*\b"),
    ("злость",      r"\b(злость|злюсь|злой|бешен|раздраж|бесит|взбеш|разъярен)\w*\b"),
    ("вина",        r"\b(вина|виноват|виновата)\w*\b"),
    ("обида",       r"\b(обид|обида|обидно|обижен|обижена)\w*\b"),
    ("грусть",      r"\b(груст|печал|тоскл|тоск|уныл|уныни)\w*\b"),
    ("радость",     r"\b(радость|рад|счастлив|доволен|довольна|восторг)\w*\b"),
    ("ступор",      r"\b(ступор|замер|оцепене)\w*\b"),
    ("отвращение",  r"\b(отвращен|противно|тошнит|мерзк|гадк)\w*\b"),
    ("надежда",     r"\b(надежд|верю|оптимизм)\w*\b"),
    ("одиночество", r"\b(одинок|одиночеств)\w*\b"),
]

EMOTION_RULES: list[tuple[re.Pattern[str], str, float, float, float, float]] = []
for _lbl, _pat in _LABEL_PATTERNS:
    _v, _a, _d, _i = _vad(_lbl)
    EMOTION_RULES.append((re.compile(_pat), _lbl, _v, _a, _d, _i))

# Special multi-word rules
_sv, _sa, _sd, _si = _vad("стыд")
EMOTION_RULES.append(
    (re.compile(r"ненавижу\s+себя|презираю\s+себя|я\s+никчем"), "стыд", _sv, _sa, _sd, _si),
)

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


# ── Context analysers (negation, intensifiers, uncertainty) ───────

def _analyze_context(text: str, match_start: int) -> tuple[bool, float, float]:
    """Analyze context around an emotion-keyword match position.

    Returns ``(is_negated, intensity_multiplier, confidence_adjustment)``.

    Priority order:
    1. Multi-word diminishers that contain negation (``не очень`` etc.) →
       treated as diminisher, **not** as negation.
    2. Simple diminishers (``немного``, ``слегка``, …) → ×0.5 intensity.
    3. Amplifiers (``очень``, ``сильно``, …) → ×1.3 intensity, +0.10 conf.
    4. Plain negation (``не``, ``нет``, …) → skip the emotion.
    5. Uncertainty markers checked globally → −0.15 conf.
    """
    before = text[:match_start].strip().lower()
    words_before = before.split()[-4:]
    chunk = " ".join(words_before)

    is_negated = False
    intensity_mult = 1.0
    conf_adj = 0.0

    # 1. Multi-word diminishers containing negation word (word-boundary match)
    if any(re.search(rf"\b{re.escape(nd)}\b", chunk) for nd in _NEGATED_DIMINISHERS):
        intensity_mult = 0.5
    # 2. Simple diminishers
    elif any(w in _DIMINISHER_WORDS for w in words_before[-3:]):
        intensity_mult = 0.5
    # 3. Amplifiers
    elif any(w in _AMPLIFIER_WORDS for w in words_before[-3:]):
        intensity_mult = 1.3
        conf_adj = 0.10
    # 4. Plain negation
    elif any(w in _NEGATION_WORDS for w in words_before[-3:]):
        is_negated = True

    # 5. Uncertainty markers (checked across the whole text)
    if any(w in _UNCERTAINTY_WORDS for w in text.lower().split()):
        conf_adj -= 0.15

    return is_negated, intensity_mult, conf_adj


def _detect_emotions(lowered: str) -> list[EmotionSignal]:
    """Layer 1: fast regex detection returning EmotionSignal objects.

    Improvements over the previous version:
    - **Negation handling**: ``я не боюсь`` no longer produces a fear signal.
    - **Intensifiers / diminutives**: ``очень грустно`` amplifies intensity;
      ``немного грустно`` dampens it.
    - **Dynamic confidence**: base 0.75 adjusted by modifiers, cause presence,
      and uncertainty markers instead of a flat 0.85.
    """
    detected: list[EmotionSignal] = []
    seen: set[str] = set()
    cause = _extract_cause(lowered)
    sarcasm = _detect_sarcasm(lowered)

    # ── "между X и Y" pattern ───────────────────────────────────
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
                    confidence=_BASE_CONFIDENCE, source="regex",
                    cause=cause, sarcasm=sarcasm,
                    multi_labels=_LABEL_TO_GOEMOTION.get(emo[0], []),
                ))

    # ── Main pattern matching with context analysis ─────────────
    for pattern, label, v, a, d, base_intensity in EMOTION_RULES:
        if label in seen:
            continue
        m = pattern.search(lowered)
        if not m:
            continue

        is_negated, intensity_mult, conf_adj = _analyze_context(lowered, m.start())
        if is_negated:
            continue  # skip negated emotions

        # Dynamic confidence
        confidence = _BASE_CONFIDENCE + conf_adj
        if cause:
            confidence += 0.05
        confidence = max(EMOTION_CONFIDENCE_MIN, min(confidence, 0.95))

        intensity = min(base_intensity * intensity_mult, 1.0)

        seen.add(label)
        detected.append(EmotionSignal(
            label=label, valence=v, arousal=a,
            dominance=d, intensity=round(intensity, 3),
            confidence=round(confidence, 3), source="regex",
            cause=cause, sarcasm=sarcasm,
            multi_labels=_LABEL_TO_GOEMOTION.get(label, []),
        ))

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
    """Layer 2: embedding-based emotion prediction with VAD interpolation.

    Instead of looking up VAD from the regex table, this version
    computes a weighted average across *all* centroids whose similarity
    exceeds 0.30 (interpolation floor) and blends the label-specific
    VAD (70 %) with the global interpolated VAD (30 %).  This gives
    actual regression from the embedding space rather than a flat
    lookup.
    """
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

    # ── compute similarities to every centroid ──────────────────
    sims: list[tuple[str, float]] = []
    for label, centroid_text in _CENTROID_TEXTS.items():
        try:
            centroid_emb = await embedding_service.embed_text(centroid_text)
            if centroid_emb is None:
                continue
        except Exception:
            continue
        sim = cosine_similarity(text_emb, centroid_emb)
        sims.append((label, sim))

    if not sims:
        return []

    sims.sort(key=lambda x: x[1], reverse=True)

    # ── global interpolated VAD from all centroids above floor ──
    relevant = [(lb, s) for lb, s in sims if s >= 0.30 and lb in _VAD_NORMS]
    total_sim = sum(s for _, s in relevant)
    if total_sim <= 0:
        return []

    v_interp = sum(_VAD_NORMS[lb][0] * s for lb, s in relevant) / total_sim
    a_interp = sum(_VAD_NORMS[lb][1] * s for lb, s in relevant) / total_sim
    d_interp = sum(_VAD_NORMS[lb][2] * s for lb, s in relevant) / total_sim

    # ── build signals for top matches above classification gate ─
    top_matches = [(lb, s) for lb, s in sims if s >= 0.45][:3]
    results: list[EmotionSignal] = []

    for label, sim in top_matches:
        confidence = min(0.3 + (sim - 0.45) * (0.65 / 0.35), 0.99)
        norms = _VAD_NORMS.get(label, (0.0, 0.0, 0.0, 0.5))

        # Blend label-specific VAD (70 %) with interpolated (30 %)
        blend = 0.7
        v = norms[0] * blend + v_interp * (1 - blend)
        a = norms[1] * blend + a_interp * (1 - blend)
        d = norms[2] * blend + d_interp * (1 - blend)

        results.append(EmotionSignal(
            label=label,
            valence=round(v, 3), arousal=round(a, 3), dominance=round(d, 3),
            intensity=norms[3],
            confidence=round(confidence, 3),
            source="model",
            multi_labels=_LABEL_TO_GOEMOTION.get(label, []),
        ))

    return results


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
        # Use dedicated arbitrate_emotion with emotion-specific prompt
        if hasattr(llm_client, "arbitrate_emotion"):
            raw = await llm_client.arbitrate_emotion(user_payload, _LLM_EMOTION_PROMPT)
        else:
            # Fallback for older clients that lack the method
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
    Also detects **ambivalence** when opposing-valence emotions
    co-occur (e.g. joy + sadness) and marks every signal accordingly.
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

    # ── Ambivalence detection ───────────────────────────────────────────────
    if len(result) >= 2:
        has_pos = any(s.valence > 0.1 for s in result)
        has_neg = any(s.valence < -0.1 for s in result)
        if has_pos and has_neg:
            for s in result:
                s.ambivalent = True

    return result


# ═══════════════════════════════════════════════════════════════════
# Guard regex
# ═══════════════════════════════════════════════════════════════════

_EMOTION_GUARD = re.compile(
    r"(боюсь|страшно|страх|тревож|тревог|беспокой|паник|нервнич|волну"
    r"|рад|радость|злюсь|злость|бешен|раздраж|бесит|взбеш|разъярен"
    r"|груст|печал|тоскл|тоск|уныл|уныни|стыд|устал|вымотан|измотан"
    r"|вина|виноват|обид|ступор|замер|оцепен"
    r"|чувствую|ненавижу\s+себя|презираю\s+себя|отвращен|противно|тошнит"
    r"|мерзк|гадк|надежд|верю|оптимизм|одинок|одиночеств"
    r"|доволен|довольна|счастлив|восторг"
    r"|плохо|хорошо|нормально|ок)",
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
