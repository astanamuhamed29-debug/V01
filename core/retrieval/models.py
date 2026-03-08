"""Data models for the identity-aware retrieval layer.

These dataclasses describe the context in which retrieval happens
(``RetrievalQueryContext``), each memory candidate being scored
(``RetrievalCandidate``), and the per-candidate score breakdown
(``RetrievalScoreBreakdown``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# ---------------------------------------------------------------------------
# Query context
# ---------------------------------------------------------------------------

#: Recognised retrieval modes that influence how candidates are weighted.
QUERY_TYPES = frozenset(
    {"chat", "planning", "proactive_action", "reflection", "goal_review"}
)


@dataclass
class RetrievalQueryContext:
    """Contextual parameters that govern a single retrieval request.

    Attributes
    ----------
    user_id:
        Identifier of the user whose memory is being searched.
    query_text:
        Raw query string (e.g. the current chat message or planning prompt).
    query_type:
        Retrieval mode.  One of ``chat``, ``planning``, ``proactive_action``,
        ``reflection``, or ``goal_review``.
    active_goals:
        List of goal identifiers (or title strings) currently active for the
        user.  Used to compute goal relevance.
    active_domains:
        Broad topic domains the user is currently focused on (e.g. ``work``,
        ``health``).
    dominant_emotions:
        Labels of the user's current dominant emotions (e.g. ``anxiety``,
        ``joy``).  Used to compute emotional salience.
    identity_signals:
        Keywords or tags that characterise the user's identity model (values,
        beliefs, dominant parts).  Used to compute identity relevance.
    confidence_threshold:
        Minimum ``confidence`` a candidate must have to be included in the
        final ranked list.  Defaults to ``0.0`` (no filtering).
    limit:
        Maximum number of results to return after ranking.  Defaults to ``10``.
    """

    user_id: str
    query_text: str
    query_type: str = "chat"
    active_goals: list[str] = field(default_factory=list)
    active_domains: list[str] = field(default_factory=list)
    dominant_emotions: list[str] = field(default_factory=list)
    identity_signals: list[str] = field(default_factory=list)
    confidence_threshold: float = 0.0
    limit: int = 10


# ---------------------------------------------------------------------------
# Retrieval candidate
# ---------------------------------------------------------------------------


@dataclass
class RetrievalCandidate:
    """A single memory node that is a candidate for inclusion in the results.

    Attributes
    ----------
    memory_id:
        Unique identifier of the memory / graph node.
    memory_type:
        Node type string (e.g. ``NOTE``, ``BELIEF``, ``VALUE``).
    content:
        Text content of the memory.
    timestamp:
        ISO-8601 creation or last-updated timestamp.  May be ``None`` if
        unavailable — scorers must degrade gracefully.
    domain:
        Broad domain tag of the memory (e.g. ``work``, ``health``).
    tags:
        Arbitrary keyword tags attached to the memory.
    embedding_score:
        Cosine similarity (or equivalent) between the query embedding and this
        node's embedding.  Range ``[0, 1]``.  Defaults to ``0.0``.
    graph_distance:
        Hop distance from the query anchor node in the knowledge graph.  ``0``
        means the node *is* the anchor; higher means more distant.  Defaults to
        ``0``.
    confidence:
        Agent confidence in the accuracy / reliability of this memory.
        Range ``[0, 1]``.  Defaults to ``1.0``.
    emotion_score:
        Emotional salience of the memory itself (e.g. VAD valence magnitude).
        Range ``[0, 1]``.  Defaults to ``0.0``.
    goal_links:
        Goal identifiers (or title strings) this memory is explicitly linked
        to.
    identity_links:
        Identity signal keywords this memory is tagged with.
    raw_payload:
        Original node payload dict for downstream consumers.
    """

    memory_id: str
    memory_type: str
    content: str
    timestamp: str | None = None
    domain: str = ""
    tags: list[str] = field(default_factory=list)
    embedding_score: float = 0.0
    graph_distance: int = 0
    confidence: float = 1.0
    emotion_score: float = 0.0
    goal_links: list[str] = field(default_factory=list)
    identity_links: list[str] = field(default_factory=list)
    raw_payload: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Score breakdown
# ---------------------------------------------------------------------------


@dataclass
class RetrievalScoreBreakdown:
    """Per-dimension score for a single ``RetrievalCandidate``.

    All individual dimension scores are in the range ``[0, 1]``.
    ``final_score`` is a weighted combination and is also clamped to
    ``[0, 1]``.

    Attributes
    ----------
    semantic_relevance:
        Contribution from embedding / textual similarity.
    goal_relevance:
        Contribution from overlap with the user's active goals.
    identity_relevance:
        Contribution from overlap with the user's identity signals.
    emotional_salience:
        Contribution from the memory's own emotional weight combined with the
        user's current dominant emotions.
    recency_score:
        Time-decay contribution — more recent memories score higher.
    confidence_score:
        Direct passthrough of the candidate's ``confidence`` field.
    relationship_score:
        Graph-distance contribution — nodes closer to the query anchor score
        higher.
    final_score:
        Weighted combination of all dimensions.
    explanation:
        Human-readable list of explanation strings for strong signals.
    """

    semantic_relevance: float = 0.0
    goal_relevance: float = 0.0
    identity_relevance: float = 0.0
    emotional_salience: float = 0.0
    recency_score: float = 0.0
    confidence_score: float = 0.0
    relationship_score: float = 0.0
    final_score: float = 0.0
    explanation: list[str] = field(default_factory=list)
