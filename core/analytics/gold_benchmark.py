from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class GoldCase:
    text: str
    expected_intent: str
    expected_emotions: list[str]
    expected_types: list[str]


@dataclass(slots=True)
class BenchmarkReport:
    samples: int
    intent_accuracy: float
    emotion_precision: float
    emotion_recall: float
    emotion_f1: float
    type_coverage: float

    def to_dict(self) -> dict:
        return {
            "samples": self.samples,
            "intent_accuracy": round(self.intent_accuracy, 4),
            "emotion_precision": round(self.emotion_precision, 4),
            "emotion_recall": round(self.emotion_recall, 4),
            "emotion_f1": round(self.emotion_f1, 4),
            "type_coverage": round(self.type_coverage, 4),
        }


def load_gold_dataset(path: str | Path) -> list[GoldCase]:
    records: list[GoldCase] = []
    with Path(path).open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            raw = json.loads(line)
            records.append(
                GoldCase(
                    text=str(raw.get("text", "")),
                    expected_intent=str(raw.get("expected_intent", "REFLECTION")).upper(),
                    expected_emotions=[str(x).lower() for x in raw.get("expected_emotions", [])],
                    expected_types=[str(x).upper() for x in raw.get("expected_types", [])],
                )
            )
    return records


async def run_gold_benchmark(cases: list[GoldCase], extractor) -> BenchmarkReport:
    """Run benchmark against async extractor callable.

    extractor signature:
      await extractor(text) -> dict with keys: intent, nodes
    """
    if not cases:
        return BenchmarkReport(0, 0.0, 0.0, 0.0, 0.0, 0.0)

    intent_ok = 0
    tp = fp = fn = 0
    type_hits = 0
    type_total = 0

    for case in cases:
        result = await extractor(case.text)
        pred_intent = str(result.get("intent", "REFLECTION")).upper()
        if pred_intent == case.expected_intent:
            intent_ok += 1

        nodes = list(result.get("nodes", []))
        pred_emotions = {
            str(n.get("metadata", {}).get("label", "")).lower()
            for n in nodes
            if str(n.get("type", "")).upper() == "EMOTION"
        }
        expected_emotions = set(case.expected_emotions)

        tp += len(pred_emotions & expected_emotions)
        fp += len(pred_emotions - expected_emotions)
        fn += len(expected_emotions - pred_emotions)

        pred_types = {str(n.get("type", "")).upper() for n in nodes}
        expected_types = set(case.expected_types)
        type_hits += len(pred_types & expected_types)
        type_total += len(expected_types)

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    return BenchmarkReport(
        samples=len(cases),
        intent_accuracy=intent_ok / len(cases),
        emotion_precision=precision,
        emotion_recall=recall,
        emotion_f1=f1,
        type_coverage=(type_hits / type_total) if type_total else 0.0,
    )
