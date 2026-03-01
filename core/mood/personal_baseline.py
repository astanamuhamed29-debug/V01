from __future__ import annotations

from dataclasses import dataclass

from core.defaults import BASELINE_MIN_SNAPSHOTS, BASELINE_WINDOW
from core.graph.model import Node
from core.graph.storage import GraphStorage


@dataclass(slots=True)
class PersonalBaselineStats:
    valence: float
    arousal: float
    dominance: float
    samples: int


class PersonalBaselineModel:
    """Persistent user baseline from mood snapshots (not global defaults)."""

    def __init__(self, storage: GraphStorage) -> None:
        self.storage = storage

    async def get(self, user_id: str) -> PersonalBaselineStats:
        snapshots = await self.storage.get_mood_snapshots(user_id, limit=BASELINE_WINDOW)
        if len(snapshots) < BASELINE_MIN_SNAPSHOTS:
            return PersonalBaselineStats(valence=0.0, arousal=0.0, dominance=0.0, samples=len(snapshots))

        vals = [float(s.get("valence_avg", 0.0)) for s in snapshots]
        ars = [float(s.get("arousal_avg", 0.0)) for s in snapshots]
        doms = [float(s.get("dominance_avg", 0.0)) for s in snapshots]

        # Robust center: trimmed mean (drop top/bottom 10%)
        def _trimmed_mean(items: list[float]) -> float:
            if not items:
                return 0.0
            xs = sorted(items)
            k = max(1, int(len(xs) * 0.1)) if len(xs) >= 10 else 0
            body = xs[k: len(xs) - k] if (len(xs) - 2 * k) > 0 else xs
            return sum(body) / len(body)

        return PersonalBaselineStats(
            valence=round(_trimmed_mean(vals), 4),
            arousal=round(_trimmed_mean(ars), 4),
            dominance=round(_trimmed_mean(doms), 4),
            samples=len(snapshots),
        )

    async def annotate_emotions(self, user_id: str, nodes: list[Node]) -> None:
        baseline = await self.get(user_id)
        for node in nodes:
            if node.type != "EMOTION":
                continue
            v = float(node.metadata.get("valence", 0.0))
            a = float(node.metadata.get("arousal", 0.0))
            d = float(node.metadata.get("dominance", 0.0))
            node.metadata["baseline_v"] = baseline.valence
            node.metadata["baseline_a"] = baseline.arousal
            node.metadata["baseline_d"] = baseline.dominance
            node.metadata["baseline_samples"] = baseline.samples
            node.metadata["delta_v_personal"] = round(v - baseline.valence, 4)
            node.metadata["delta_a_personal"] = round(a - baseline.arousal, 4)
            node.metadata["delta_d_personal"] = round(d - baseline.dominance, 4)
