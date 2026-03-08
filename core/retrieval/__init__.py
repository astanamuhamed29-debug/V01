"""Identity-aware retrieval layer for SELF-OS.

Provides explainable, multi-dimensional scoring and ranking of memory
candidates beyond pure semantic similarity.
"""

from core.retrieval.models import (
    RetrievalCandidate,
    RetrievalQueryContext,
    RetrievalScoreBreakdown,
)
from core.retrieval.ranker import RetrievalRanker
from core.retrieval.scoring import RetrievalScorer

__all__ = [
    "RetrievalCandidate",
    "RetrievalQueryContext",
    "RetrievalRanker",
    "RetrievalScoreBreakdown",
    "RetrievalScorer",
]
