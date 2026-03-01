"""IdentitySnapshot — unified digital fingerprint of a person's psychological identity.

Aggregates all pattern-analysis layers into a single snapshot:

1. **EmotionalCore** — baseline VAD, emotion distribution, volatility,
   reactivity, recovery speed, dominant axis, ambivalence ratio, top triggers.

2. **Cross-pattern correlations** — emotion↔cognition, part↔emotion,
   emotion↔need, trigger→emotion→part chains discovered via temporal
   co-occurrence analysis.

3. **Core identity facets** — beliefs, values, needs, parts, cognitive
   style — ranked by centrality / signal strength.

4. **Change trajectory** — improving / declining / stable trend with
   deltas computed from mood snapshots over the last 30 days.

Usage::

    builder = IdentitySnapshotBuilder(storage, embedding_service)
    snapshot = await builder.build(user_id)
    # snapshot.emotional_core.volatility → 0.32
    # snapshot.correlations[0] → emotion-cognition link
"""

from __future__ import annotations

import logging
import math
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any

from core.analytics.pattern_analyzer import PatternAnalyzer, PatternReport
from core.graph.model import Node, edge_weight
from core.graph.storage import GraphStorage

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════
# Data structures
# ═══════════════════════════════════════════════════════════════════

@dataclass(slots=True)
class EmotionalCore:
    """The emotional nucleus — how this person fundamentally feels."""

    # Baseline VAD (what is "neutral" for them — EMA from all EMOTION nodes)
    baseline_valence: float = 0.0
    baseline_arousal: float = 0.0
    baseline_dominance: float = 0.0

    # Distribution of emotions (label → frequency ratio, sums to 1.0)
    emotion_distribution: dict[str, float] = field(default_factory=dict)

    # Emotional volatility (std dev of valence across time)
    volatility: float = 0.0

    # Reactivity (mean intensity of emotional signals)
    reactivity: float = 0.0

    # Recovery speed: how fast valence returns to baseline after
    # a negative dip.  Higher = recovers faster.  0 = insufficient data.
    recovery_speed: float = 0.0

    # Dominant emotional axis — which VAD dimension varies most
    dominant_axis: str = "valence"  # "valence" | "arousal" | "dominance"

    # Ratio of ambivalent signals (opposing valences co-occurring)
    ambivalence_ratio: float = 0.0

    # Top 5 emotional triggers (source_text, target_emotion, strength)
    top_triggers: list[dict[str, Any]] = field(default_factory=list)

    # Total emotion nodes analyzed
    sample_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "baseline_valence": round(self.baseline_valence, 3),
            "baseline_arousal": round(self.baseline_arousal, 3),
            "baseline_dominance": round(self.baseline_dominance, 3),
            "emotion_distribution": {
                k: round(v, 3) for k, v in self.emotion_distribution.items()
            },
            "volatility": round(self.volatility, 3),
            "reactivity": round(self.reactivity, 3),
            "recovery_speed": round(self.recovery_speed, 3),
            "dominant_axis": self.dominant_axis,
            "ambivalence_ratio": round(self.ambivalence_ratio, 3),
            "top_triggers": self.top_triggers[:5],
            "sample_count": self.sample_count,
        }


@dataclass(slots=True)
class CorrelationCluster:
    """A discovered cross-modal correlation between pattern types."""

    pattern_a_type: str   # "emotion" | "cognition" | "part" | "need"
    pattern_a_label: str
    pattern_b_type: str
    pattern_b_label: str
    co_occurrence_count: int
    co_occurrence_score: float   # 0..1 — Jaccard-like
    direction: str = "bidirectional"  # "a→b" | "b→a" | "bidirectional"

    def to_dict(self) -> dict[str, Any]:
        return {
            "a": f"{self.pattern_a_type}:{self.pattern_a_label}",
            "b": f"{self.pattern_b_type}:{self.pattern_b_label}",
            "co_occurrence": self.co_occurrence_count,
            "score": round(self.co_occurrence_score, 3),
            "direction": self.direction,
        }


@dataclass(slots=True)
class IdentitySnapshot:
    """Complete digital fingerprint of a person's psychological identity."""

    user_id: str
    generated_at: str
    version: int = 1

    # ── Emotional Core ──────────────────────────────────────────
    emotional_core: EmotionalCore = field(default_factory=EmotionalCore)

    # ── Core beliefs (highest PageRank / most referenced) ───────
    core_beliefs: list[dict[str, Any]] = field(default_factory=list)

    # ── Core values (ranked by appearances) ─────────────────────
    core_values: list[dict[str, Any]] = field(default_factory=list)

    # ── Active needs (sorted by total_signals) ──────────────────
    active_needs: list[dict[str, Any]] = field(default_factory=list)

    # ── Part system snapshot (IFS) ──────────────────────────────
    part_system: list[dict[str, Any]] = field(default_factory=list)

    # ── Cognitive style (dominant distortions) ──────────────────
    cognitive_style: list[dict[str, Any]] = field(default_factory=list)

    # ── Cross-pattern correlations ──────────────────────────────
    correlations: list[CorrelationCluster] = field(default_factory=list)

    # ── Syndromes (dense graph clusters) ────────────────────────
    syndromes: list[dict[str, Any]] = field(default_factory=list)

    # ── Data depth metrics ──────────────────────────────────────
    data_depth: dict[str, Any] = field(default_factory=dict)

    # ── Change trajectory ───────────────────────────────────────
    trajectory: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "user_id": self.user_id,
            "generated_at": self.generated_at,
            "version": self.version,
            "emotional_core": self.emotional_core.to_dict(),
            "core_beliefs": self.core_beliefs,
            "core_values": self.core_values,
            "active_needs": self.active_needs,
            "part_system": self.part_system,
            "cognitive_style": self.cognitive_style,
            "correlations": [c.to_dict() for c in self.correlations],
            "syndromes": self.syndromes,
            "data_depth": self.data_depth,
            "trajectory": self.trajectory,
        }


# ═══════════════════════════════════════════════════════════════════
# Builder
# ═══════════════════════════════════════════════════════════════════

class IdentitySnapshotBuilder:
    """Computes a full IdentitySnapshot from the graph storage.

    Parameters
    ----------
    storage : GraphStorage
    embedding_service : optional
        When provided, PatternAnalyzer gets semantic implicit-link detection.
    """

    def __init__(
        self,
        storage: GraphStorage,
        embedding_service: Any | None = None,
    ) -> None:
        self.storage = storage
        self.embedding_service = embedding_service
        self._pattern_analyzer = PatternAnalyzer(storage, embedding_service)

    async def build(self, user_id: str, days: int = 30) -> IdentitySnapshot:
        """Build a complete identity snapshot for *user_id*.

        Fetches all relevant data in parallel, computes emotional core,
        cross-pattern correlations, and packages everything into a single
        :class:`IdentitySnapshot`.
        """
        import asyncio

        # ── Parallel data fetch ─────────────────────────────────
        pattern_report_task = self._pattern_analyzer.analyze(user_id, days=days)
        emotion_nodes_task = self.storage.find_nodes_recent(
            user_id=user_id, node_type="EMOTION", limit=500,
        )
        mood_snapshots_task = self.storage.get_mood_snapshots(user_id, limit=30)
        beliefs_task = self.storage.find_nodes(user_id, node_type="BELIEF", limit=100)
        values_task = self.storage.find_nodes(user_id, node_type="VALUE", limit=50)
        thoughts_task = self.storage.find_nodes_recent(user_id, "THOUGHT", limit=300)
        parts_task = self.storage.find_nodes(user_id, node_type="PART", limit=50)
        all_edges_task = self.storage.list_edges(user_id)

        (
            pattern_report,
            emotion_nodes,
            mood_snapshots,
            belief_nodes,
            value_nodes,
            thought_nodes,
            part_nodes,
            all_edges,
        ) = await asyncio.gather(
            pattern_report_task,
            emotion_nodes_task,
            mood_snapshots_task,
            beliefs_task,
            values_task,
            thoughts_task,
            parts_task,
            all_edges_task,
        )

        # ── Compute emotional core ──────────────────────────────
        emotional_core = self._compute_emotional_core(
            emotion_nodes, mood_snapshots, pattern_report,
        )

        # ── Compute cross-pattern correlations ──────────────────
        correlations = self._compute_correlations(
            emotion_nodes, thought_nodes, part_nodes, all_edges,
        )

        # ── Core beliefs ────────────────────────────────────────
        core_beliefs = self._rank_beliefs(belief_nodes)

        # ── Core values ─────────────────────────────────────────
        core_values = self._rank_values(value_nodes)

        # ── Active needs ────────────────────────────────────────
        active_needs = [
            {
                "name": n.need_name,
                "signals": n.total_signals,
                "emotions": n.emotions_signaling,
                "parts": n.parts_protecting,
            }
            for n in pattern_report.need_profile[:5]
        ]

        # ── Part system ─────────────────────────────────────────
        part_system = [
            {
                "name": p.part_name,
                "subtype": p.subtype,
                "appearances": p.appearances,
                "trend": p.trend,
                "dominant_need": p.dominant_need,
                "voice": p.voice,
            }
            for p in pattern_report.part_dynamics
        ]

        # ── Cognitive style ─────────────────────────────────────
        cognitive_style = [
            {
                "distortion": c.distortion_ru,
                "distortion_en": c.distortion,
                "count": c.count,
                "example": c.example_thought[:120],
            }
            for c in pattern_report.cognition_patterns[:5]
        ]

        # ── Syndromes ───────────────────────────────────────────
        syndromes = [
            {"nodes": s.nodes, "core_theme": s.core_theme, "density": round(s.score, 3)}
            for s in pattern_report.syndromes[:3]
        ]

        # ── Data depth metrics ──────────────────────────────────
        total_nodes = (
            len(emotion_nodes) + len(belief_nodes) + len(value_nodes)
            + len(thought_nodes) + len(part_nodes)
        )
        data_depth = {
            "emotion_count": len(emotion_nodes),
            "belief_count": len(belief_nodes),
            "value_count": len(value_nodes),
            "thought_count": len(thought_nodes),
            "part_count": len(part_nodes),
            "edge_count": len(all_edges),
            "mood_snapshots": len(mood_snapshots),
            "total_nodes": total_nodes,
            "analysis_days": days,
            "has_enough_data": total_nodes > 10,
        }

        # ── Trajectory ──────────────────────────────────────────
        trajectory = self._compute_trajectory(mood_snapshots)

        return IdentitySnapshot(
            user_id=user_id,
            generated_at=datetime.now(timezone.utc).isoformat(),
            emotional_core=emotional_core,
            core_beliefs=core_beliefs,
            core_values=core_values,
            active_needs=active_needs,
            part_system=part_system,
            cognitive_style=cognitive_style,
            correlations=correlations,
            syndromes=syndromes,
            data_depth=data_depth,
            trajectory=trajectory,
        )

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Emotional Core computation
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def _compute_emotional_core(
        self,
        emotion_nodes: list[Node],
        mood_snapshots: list[dict],
        pattern_report: PatternReport,
    ) -> EmotionalCore:
        if not emotion_nodes:
            return EmotionalCore()

        # ── Baseline VAD (weighted EMA from all emotion nodes) ──
        valences: list[float] = []
        arousals: list[float] = []
        dominances: list[float] = []
        intensities: list[float] = []
        labels: list[str] = []
        ambivalent_count = 0

        now = datetime.now(timezone.utc)
        weights: list[float] = []

        for node in emotion_nodes:
            v = float(node.metadata.get("valence", 0))
            a = float(node.metadata.get("arousal", 0))
            d = float(node.metadata.get("dominance", 0))
            i = float(node.metadata.get("intensity", 0.5))
            conf = float(node.metadata.get("confidence", 0.7))

            valences.append(v)
            arousals.append(a)
            dominances.append(d)
            intensities.append(i)

            label = str(node.metadata.get("label", ""))
            if label:
                labels.append(label)

            if node.metadata.get("ambivalent"):
                ambivalent_count += 1

            # temporal weight
            ts = node.metadata.get("created_at") or node.created_at
            try:
                dt = datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
            except (ValueError, TypeError):
                dt = now
            age_days = max((now - dt).total_seconds() / 86400.0, 0.0)
            w = conf * i / (1.0 + age_days / 14.0)
            weights.append(w)

        total_w = sum(weights) or 1.0
        baseline_v = sum(v * w for v, w in zip(valences, weights)) / total_w
        baseline_a = sum(a * w for a, w in zip(arousals, weights)) / total_w
        baseline_d = sum(d * w for d, w in zip(dominances, weights)) / total_w

        # ── Emotion distribution ────────────────────────────────
        label_counts = Counter(labels)
        total_labels = sum(label_counts.values()) or 1
        distribution = {
            label: count / total_labels
            for label, count in label_counts.most_common(10)
        }

        # ── Volatility (std dev of valence) ─────────────────────
        volatility = statistics.pstdev(valences) if len(valences) >= 2 else 0.0

        # ── Reactivity (mean intensity) ─────────────────────────
        reactivity = sum(intensities) / len(intensities) if intensities else 0.0

        # ── Recovery speed ──────────────────────────────────────
        recovery_speed = self._compute_recovery_speed(mood_snapshots)

        # ── Dominant axis ───────────────────────────────────────
        v_std = statistics.pstdev(valences) if len(valences) >= 2 else 0.0
        a_std = statistics.pstdev(arousals) if len(arousals) >= 2 else 0.0
        d_std = statistics.pstdev(dominances) if len(dominances) >= 2 else 0.0
        axis_map = {"valence": v_std, "arousal": a_std, "dominance": d_std}
        dominant_axis = max(axis_map, key=lambda k: axis_map[k])

        # ── Ambivalence ratio ───────────────────────────────────
        ambivalence_ratio = ambivalent_count / len(emotion_nodes) if emotion_nodes else 0.0

        # ── Top triggers ────────────────────────────────────────
        top_triggers: list[dict[str, Any]] = []
        for tp in pattern_report.trigger_patterns[:5]:
            if tp.target_type == "EMOTION":
                top_triggers.append({
                    "trigger": tp.source_text[:60],
                    "trigger_type": tp.source_type,
                    "emotion": tp.target_name,
                    "occurrences": tp.occurrences,
                    "strength": round(tp.weighted_score, 2),
                })

        return EmotionalCore(
            baseline_valence=baseline_v,
            baseline_arousal=baseline_a,
            baseline_dominance=baseline_d,
            emotion_distribution=distribution,
            volatility=volatility,
            reactivity=reactivity,
            recovery_speed=recovery_speed,
            dominant_axis=dominant_axis,
            ambivalence_ratio=ambivalence_ratio,
            top_triggers=top_triggers,
            sample_count=len(emotion_nodes),
        )

    def _compute_recovery_speed(self, snapshots: list[dict]) -> float:
        """Compute recovery speed from mood snapshots.

        Looks for "dip → recovery" patterns: a snapshot with valence < -0.3
        followed by a return above -0.1.  Recovery speed is inversely
        proportional to the number of snapshots it takes to recover.
        """
        if len(snapshots) < 3:
            return 0.0

        # snapshots are newest-first — reverse for chronological order
        ordered = list(reversed(snapshots))
        recovery_episodes: list[float] = []

        i = 0
        while i < len(ordered) - 1:
            v = float(ordered[i].get("valence_avg", 0))
            if v < -0.3:
                # Found a dip — measure how many steps to recover
                steps = 0
                for j in range(i + 1, len(ordered)):
                    steps += 1
                    v_next = float(ordered[j].get("valence_avg", 0))
                    if v_next > -0.1:
                        # Recovered
                        speed = 1.0 / steps
                        recovery_episodes.append(speed)
                        i = j
                        break
                else:
                    # Didn't recover in the window
                    recovery_episodes.append(0.0)
            i += 1

        if not recovery_episodes:
            return 0.0
        return sum(recovery_episodes) / len(recovery_episodes)

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Cross-pattern correlation engine
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def _compute_correlations(
        self,
        emotion_nodes: list[Node],
        thought_nodes: list[Node],
        part_nodes: list[Node],
        all_edges: list,
    ) -> list[CorrelationCluster]:
        """Discover cross-modal correlations via temporal co-occurrence.

        Nodes created within the same 1-hour window are considered
        co-occurring.  Jaccard similarity between the occurrence sets
        of each label pair gives the correlation score.
        """
        # ── Build time-bucketed occurrence sets ─────────────────
        # bucket_key = ISO date + hour (e.g. "2025-03-01T14")
        BUCKET_FMT = "%Y-%m-%dT%H"

        def _bucket(node: Node) -> str:
            ts = node.metadata.get("created_at") or node.created_at
            try:
                dt = datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
                return dt.strftime(BUCKET_FMT)
            except (ValueError, TypeError):
                return "unknown"

        def _label(node: Node, ntype: str) -> str | None:
            if ntype == "emotion":
                return str(node.metadata.get("label", "")).strip() or None
            if ntype == "cognition":
                return str(node.metadata.get("distortion", "")).strip() or None
            if ntype == "part":
                return (node.name or node.subtype or node.key or "").strip() or None
            return None

        # Collect labelled occurrences: type:label → set of buckets
        occurrence_sets: dict[tuple[str, str], set[str]] = defaultdict(set)

        for node in emotion_nodes:
            label = _label(node, "emotion")
            if label:
                occurrence_sets[("emotion", label)].add(_bucket(node))

        for node in thought_nodes:
            label = _label(node, "cognition")
            if label:
                occurrence_sets[("cognition", label)].add(_bucket(node))

        for node in part_nodes:
            label = _label(node, "part")
            if label:
                occurrence_sets[("part", label)].add(_bucket(node))

        # ── Compute Jaccard between cross-modal pairs ───────────
        keys = list(occurrence_sets.keys())
        correlations: list[CorrelationCluster] = []

        for i in range(len(keys)):
            for j in range(i + 1, len(keys)):
                type_a, label_a = keys[i]
                type_b, label_b = keys[j]

                # Only cross-modal correlations (not emotion↔emotion)
                if type_a == type_b:
                    continue

                set_a = occurrence_sets[keys[i]]
                set_b = occurrence_sets[keys[j]]
                intersection = set_a & set_b
                union = set_a | set_b

                if not union or len(intersection) < 2:
                    continue

                jaccard = len(intersection) / len(union)
                if jaccard < 0.15:
                    continue

                # Direction heuristic: if one always appears before the other
                direction = "bidirectional"

                correlations.append(CorrelationCluster(
                    pattern_a_type=type_a,
                    pattern_a_label=label_a,
                    pattern_b_type=type_b,
                    pattern_b_label=label_b,
                    co_occurrence_count=len(intersection),
                    co_occurrence_score=jaccard,
                    direction=direction,
                ))

        # Also add edge-based correlations for NEED signals
        need_emotion_links = self._edge_based_need_correlations(all_edges)
        correlations.extend(need_emotion_links)

        # Deduplicate and sort
        seen: set[frozenset[str]] = set()
        unique: list[CorrelationCluster] = []
        for c in correlations:
            key = frozenset({f"{c.pattern_a_type}:{c.pattern_a_label}",
                             f"{c.pattern_b_type}:{c.pattern_b_label}"})
            if key not in seen:
                seen.add(key)
                unique.append(c)

        unique.sort(key=lambda c: c.co_occurrence_score, reverse=True)
        return unique[:15]

    def _edge_based_need_correlations(
        self,
        all_edges: list,
    ) -> list[CorrelationCluster]:
        """Extract emotion↔need correlations from SIGNALS_NEED edges."""
        # Count co-occurrences of emotion→need via SIGNALS_NEED edges
        # We don't have node data here easily, so this is a placeholder
        # that returns based on edge relation patterns
        # (the actual data is already captured in PatternReport.need_profile)
        return []

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Facet ranking
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def _rank_beliefs(self, belief_nodes: list[Node]) -> list[dict[str, Any]]:
        """Rank beliefs by salience, revision count, and recency."""
        scored: list[tuple[float, Node]] = []
        now = datetime.now(timezone.utc)
        for node in belief_nodes:
            salience = float(node.metadata.get("salience_score", 0.5))
            revisions = int(node.metadata.get("revision_count", 0))
            try:
                dt = datetime.fromisoformat(node.created_at.replace("Z", "+00:00"))
            except (ValueError, TypeError):
                dt = now
            age_days = max((now - dt).total_seconds() / 86400.0, 0.0)
            recency = 1.0 / (1.0 + age_days / 30.0)
            score = salience * (1 + revisions * 0.3) * recency
            scored.append((score, node))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [
            {
                "text": (n.text or n.name or "")[:200],
                "salience": round(float(n.metadata.get("salience_score", 0.5)), 2),
                "revisions": int(n.metadata.get("revision_count", 0)),
                "score": round(score, 3),
            }
            for score, n in scored[:7]
        ]

    def _rank_values(self, value_nodes: list[Node]) -> list[dict[str, Any]]:
        """Rank values by appearances count."""
        ranked = sorted(
            value_nodes,
            key=lambda n: int(n.metadata.get("appearances", 1)),
            reverse=True,
        )
        return [
            {
                "name": n.name or n.key or "",
                "appearances": int(n.metadata.get("appearances", 1)),
            }
            for n in ranked[:7]
        ]

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Change trajectory
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def _compute_trajectory(self, snapshots: list[dict]) -> dict[str, Any]:
        """Compute change trajectory from mood snapshots.

        Compares first half vs second half of the snapshot window:
        - trend: "improving" | "declining" | "stable"
        - delta_valence, delta_arousal, delta_dominance
        - recent_dominant_label
        """
        if len(snapshots) < 2:
            return {
                "trend": "unknown",
                "delta_valence": 0.0,
                "delta_arousal": 0.0,
                "delta_dominance": 0.0,
                "recent_dominant_label": None,
                "snapshot_count": len(snapshots),
            }

        # snapshots are newest-first
        mid = len(snapshots) // 2
        recent = snapshots[:mid] or snapshots[:1]
        older = snapshots[mid:] or snapshots[-1:]

        def _avg(snaps: list[dict], key: str) -> float:
            vals = [float(s.get(key, 0)) for s in snaps]
            return sum(vals) / len(vals) if vals else 0.0

        recent_v = _avg(recent, "valence_avg")
        older_v = _avg(older, "valence_avg")
        recent_a = _avg(recent, "arousal_avg")
        older_a = _avg(older, "arousal_avg")
        recent_d = _avg(recent, "dominance_avg")
        older_d = _avg(older, "dominance_avg")

        dv = recent_v - older_v
        da = recent_a - older_a
        dd = recent_d - older_d

        if dv > 0.15:
            trend = "improving"
        elif dv < -0.15:
            trend = "declining"
        else:
            trend = "stable"

        recent_label = snapshots[0].get("dominant_label") if snapshots else None

        return {
            "trend": trend,
            "delta_valence": round(dv, 3),
            "delta_arousal": round(da, 3),
            "delta_dominance": round(dd, 3),
            "recent_dominant_label": recent_label,
            "snapshot_count": len(snapshots),
        }
