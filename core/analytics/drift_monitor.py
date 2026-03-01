from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from core.defaults import (
    DRIFT_AROUSAL_SHIFT_WARN,
    DRIFT_JS_WARN,
    DRIFT_MIN_SNAPSHOTS,
    DRIFT_VALENCE_SHIFT_WARN,
)
from core.graph.storage import GraphStorage


@dataclass(slots=True)
class DriftReport:
    alert: bool
    severity: str
    valence_shift: float
    arousal_shift: float
    js_divergence: float
    samples: int

    def to_dict(self) -> dict:
        return {
            "alert": self.alert,
            "severity": self.severity,
            "valence_shift": round(self.valence_shift, 4),
            "arousal_shift": round(self.arousal_shift, 4),
            "js_divergence": round(self.js_divergence, 4),
            "samples": self.samples,
        }


def _js_divergence(p: dict[str, float], q: dict[str, float]) -> float:
    import math

    labels = set(p) | set(q)
    if not labels:
        return 0.0

    def _norm(d: dict[str, float]) -> dict[str, float]:
        total = sum(d.values()) or 1.0
        return {k: v / total for k, v in d.items()}

    p = _norm(p)
    q = _norm(q)
    m = {k: 0.5 * p.get(k, 0.0) + 0.5 * q.get(k, 0.0) for k in labels}

    def _kl(a: dict[str, float], b: dict[str, float]) -> float:
        eps = 1e-12
        s = 0.0
        for k in labels:
            av = max(a.get(k, 0.0), eps)
            bv = max(b.get(k, 0.0), eps)
            s += av * math.log(av / bv)
        return s

    return 0.5 * _kl(p, m) + 0.5 * _kl(q, m)


class DriftMonitor:
    """Monitors extraction drift from mood and emotion distributions."""

    def __init__(self, storage: GraphStorage) -> None:
        self.storage = storage

    async def evaluate(self, user_id: str) -> DriftReport:
        snapshots = await self.storage.get_mood_snapshots(user_id, limit=24)
        if len(snapshots) < DRIFT_MIN_SNAPSHOTS:
            return DriftReport(False, "none", 0.0, 0.0, 0.0, len(snapshots))

        half = len(snapshots) // 2
        recent = snapshots[:half]
        baseline = snapshots[half:]

        recent_valence = sum(float(x.get("valence_avg", 0.0)) for x in recent) / max(len(recent), 1)
        baseline_valence = sum(float(x.get("valence_avg", 0.0)) for x in baseline) / max(len(baseline), 1)
        recent_arousal = sum(float(x.get("arousal_avg", 0.0)) for x in recent) / max(len(recent), 1)
        baseline_arousal = sum(float(x.get("arousal_avg", 0.0)) for x in baseline) / max(len(baseline), 1)

        val_shift = recent_valence - baseline_valence
        ar_shift = recent_arousal - baseline_arousal

        recent_counts = Counter(str(x.get("dominant_label") or "") for x in recent if x.get("dominant_label"))
        baseline_counts = Counter(str(x.get("dominant_label") or "") for x in baseline if x.get("dominant_label"))
        js = _js_divergence(dict(recent_counts), dict(baseline_counts))

        warn = (
            abs(val_shift) >= DRIFT_VALENCE_SHIFT_WARN
            or abs(ar_shift) >= DRIFT_AROUSAL_SHIFT_WARN
            or js >= DRIFT_JS_WARN
        )
        if not warn:
            severity = "none"
        elif js >= DRIFT_JS_WARN * 1.7 or abs(val_shift) >= DRIFT_VALENCE_SHIFT_WARN * 1.7:
            severity = "high"
        else:
            severity = "medium"

        return DriftReport(
            alert=warn,
            severity=severity,
            valence_shift=val_shift,
            arousal_shift=ar_shift,
            js_divergence=js,
            samples=len(snapshots),
        )
