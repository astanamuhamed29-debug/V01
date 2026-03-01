"""Cognitive Distortion Detector for SELF-OS.

Detects common cognitive distortions in user messages using regex + keyword
patterns and produces structured results with reframe suggestions.

Detected distortion types
-------------------------
* ``CATASTROPHIZING`` — catastrophic thinking
* ``BLACK_WHITE`` — all-or-nothing thinking
* ``MIND_READING`` — assuming others' thoughts
* ``PERSONALIZATION`` — self-blaming
* ``OVERGENERALIZATION`` — over-generalising from single events
* ``EMOTIONAL_REASONING`` — treating feelings as facts
* ``SHOULD_STATEMENTS`` — rigid rules about what one must do
* ``LABELING`` — global negative self-labels
"""

from __future__ import annotations

import re
from dataclasses import dataclass

__all__ = ["CognitiveDistortionDetector", "CognitiveDistortion"]


@dataclass
class CognitiveDistortion:
    """A single detected cognitive distortion instance."""

    distortion_type: str
    confidence: float
    evidence_text: str
    reframe_suggestion: str


# ---------------------------------------------------------------------------
# Pattern definitions
# ---------------------------------------------------------------------------

_PATTERNS: list[dict] = [
    {
        "type": "CATASTROPHIZING",
        "patterns": [
            r"\bконец\b",
            r"\bникогда\b.*\bне\b",
            r"\bвсё\s+пропал",
            r"\bкатастроф",
            r"\bужас",
            r"\bстрашн",
            r"\bнеизбежно\b",
        ],
        "reframe": (
            "Попробуй рассмотреть более реалистичный сценарий. "
            "Каков наиболее вероятный исход, а не наихудший?"
        ),
    },
    {
        "type": "BLACK_WHITE",
        "patterns": [
            r"\bвсегда\b",
            r"\bникогда\b",
            r"\bвсе\b.*\b(плохо|хорошо)\b",
            r"\bникто\b",
            r"\bтолько\b.*\b(я|мне|меня)\b",
            r"\bабсолютно\b",
            r"\bсовсем\b.*\bне\b",
        ],
        "reframe": (
            "Реальность редко бывает чёрно-белой. "
            "Есть ли промежуточные варианты или исключения?"
        ),
    },
    {
        "type": "MIND_READING",
        "patterns": [
            r"\bони\s+думают\b",
            r"\bон\s+считает\s+что\s+я\b",
            r"\bвсе\s+знают\s+что\b",
            r"\bонa?\s+думает\b",
            r"\bони\s+считают\b",
            r"\bнаверняка\s+думают\b",
        ],
        "reframe": (
            "Ты не можешь знать наверняка, что думают другие. "
            "Есть ли у тебя прямые доказательства этой мысли?"
        ),
    },
    {
        "type": "PERSONALIZATION",
        "patterns": [
            r"\bиз-за\s+меня\b",
            r"\bмоя\s+вина\b",
            r"\bя\s+виноват",
            r"\bя\s+виновна",
            r"\bэто\s+я\s+виноват",
            r"\bя\s+испортил",
            r"\bя\s+сломал",
        ],
        "reframe": (
            "Является ли это действительно только твоей ответственностью? "
            "Какие другие факторы могли повлиять на ситуацию?"
        ),
    },
    {
        "type": "OVERGENERALIZATION",
        "patterns": [
            r"\bкаждый\s+раз\b",
            r"\bпостоянно\b",
            r"\bвечно\b",
            r"\bвсё\s+время\b",
            r"\bснова\s+и\s+снова\b",
            r"\bопять\s+то\s+же\b",
        ],
        "reframe": (
            "Так ли это происходит каждый раз? "
            "Можешь ли ты вспомнить случаи, когда всё шло иначе?"
        ),
    },
    {
        "type": "EMOTIONAL_REASONING",
        "patterns": [
            r"\bя\s+чувствую\b.*\bзначит\s+это\s+правда\b",
            r"\bраз\s+мне\s+плохо\s+значит\b",
            r"\bя\s+чувствую\s+что\s+(я\s+)?(плохой|неудачник|слабый)\b",
            r"\bмне\s+кажется.*\bзначит\s+так\s+и\s+есть\b",
        ],
        "reframe": (
            "Чувства — это не факты. "
            "Какие объективные доказательства подтверждают или опровергают эту мысль?"
        ),
    },
    {
        "type": "SHOULD_STATEMENTS",
        "patterns": [
            r"\bдолжен\b",
            r"\bдолжна\b",
            r"\bобязан\b",
            r"\bобязана\b",
            r"\bне\s+имею\s+права\b",
            r"\bнельзя\s+мне\b",
            r"\bя\s+обязательно\s+должен\b",
        ],
        "reframe": (
            "Слова «должен» и «обязан» создают жёсткие правила. "
            "Что будет, если заменить «должен» на «хотел бы»?"
        ),
    },
    {
        "type": "LABELING",
        "patterns": [
            r"\bя\s+неудачник\b",
            r"\bя\s+(такой\s+)?тупой\b",
            r"\bя\s+слабый\b",
            r"\bя\s+слабая\b",
            r"\bя\s+лузер\b",
            r"\bя\s+ничтожество\b",
            r"\bя\s+бесполезный\b",
            r"\bя\s+плохой\s+человек\b",
        ],
        "reframe": (
            "Ты описываешь поступок или ситуацию, а не свою сущность. "
            "Какое более точное и справедливое описание ты мог бы дать?"
        ),
    },
]

# Pre-compile all patterns for performance
from core.defaults import COGNITIVE_CONFIDENCE_BASELINE as _CONFIDENCE_BASELINE
_COMPILED: list[dict] = [
    {
        **entry,
        "_compiled": [re.compile(p, re.IGNORECASE | re.UNICODE) for p in entry["patterns"]],
    }
    for entry in _PATTERNS
]


# ---------------------------------------------------------------------------
# Detector
# ---------------------------------------------------------------------------


class CognitiveDistortionDetector:
    """Detect cognitive distortions in a piece of text.

    Usage::

        detector = CognitiveDistortionDetector()
        distortions = detector.detect("Я всегда всё порчу, это моя вина.")
    """

    def detect(self, text: str) -> list[CognitiveDistortion]:
        """Return a list of detected :class:`CognitiveDistortion` instances.

        Parameters
        ----------
        text:
            The user message to analyse.
        """
        results: list[CognitiveDistortion] = []
        for entry in _COMPILED:
            matches: list[str] = []
            for pattern in entry["_compiled"]:
                for m in pattern.finditer(text):
                    matches.append(m.group(0))
            if matches:
                # Confidence proportional to number of matching patterns
                confidence = min(1.0, len(matches) / len(entry["_compiled"]) + _CONFIDENCE_BASELINE)
                evidence = "; ".join(dict.fromkeys(matches))  # deduplicate, preserve order
                results.append(
                    CognitiveDistortion(
                        distortion_type=entry["type"],
                        confidence=round(confidence, 2),
                        evidence_text=evidence,
                        reframe_suggestion=entry["reframe"],
                    )
                )
        return results
