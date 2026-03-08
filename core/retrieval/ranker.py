"""Retrieval ranker for SELF-OS.

``RetrievalRanker`` orchestrates the full retrieval pipeline:

1. Score each candidate using :class:`~core.retrieval.scoring.RetrievalScorer`.
2. Filter out candidates whose ``confidence`` falls below
   ``context.confidence_threshold``.
3. Sort the remaining candidates descending by ``final_score``.
4. Return up to ``context.limit`` results, each paired with its
   :class:`~core.retrieval.models.RetrievalScoreBreakdown`.
"""

from __future__ import annotations

from dataclasses import dataclass

from core.retrieval.models import (
    RetrievalCandidate,
    RetrievalQueryContext,
    RetrievalScoreBreakdown,
)
from core.retrieval.scoring import RetrievalScorer


@dataclass
class RankedResult:
    """A candidate together with its full score breakdown.

    Attributes
    ----------
    candidate:
        The original memory candidate.
    breakdown:
        The per-dimension score breakdown produced by
        :class:`~core.retrieval.scoring.RetrievalScorer`.
    """

    candidate: RetrievalCandidate
    breakdown: RetrievalScoreBreakdown


class RetrievalRanker:
    """Scores, filters, and ranks memory candidates.

    Parameters
    ----------
    scorer:
        Optional custom :class:`~core.retrieval.scoring.RetrievalScorer`
        instance.  If not provided, a default scorer with v0 weights is
        created automatically.
    """

    def __init__(self, scorer: RetrievalScorer | None = None) -> None:
        self._scorer = scorer if scorer is not None else RetrievalScorer()

    def rank(
        self,
        candidates: list[RetrievalCandidate],
        context: RetrievalQueryContext,
    ) -> list[RankedResult]:
        """Score, filter, sort, and cap *candidates* for *context*.

        Parameters
        ----------
        candidates:
            Unordered list of memory candidates to evaluate.
        context:
            Query context providing goals, identity signals, emotions, and
            retrieval limits.

        Returns
        -------
        list[RankedResult]
            Up to ``context.limit`` results sorted descending by
            ``final_score``.  Each result exposes the candidate alongside its
            full score breakdown and explanation.
        """
        scored: list[RankedResult] = []

        for candidate in candidates:
            # Skip candidates that do not meet the minimum confidence bar.
            if candidate.confidence < context.confidence_threshold:
                continue

            breakdown = self._scorer.score(candidate, context)
            scored.append(RankedResult(candidate=candidate, breakdown=breakdown))

        # Sort descending by final score; use memory_id as a stable tie-breaker.
        scored.sort(
            key=lambda r: (r.breakdown.final_score, r.candidate.memory_id),
            reverse=True,
        )

        return scored[: context.limit]
