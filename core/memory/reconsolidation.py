"""ReconsolidationEngine — detects and handles belief contradictions.

When a user expresses something that contradicts an existing BELIEF node,
the engine flags it and can update the belief with the new evidence.

Detection uses cosine similarity between the new text embedding and
existing BELIEF embeddings:
* similarity ∈ [0.5, 0.75] → potential contradiction (semantically related
  but different).
* A future LLM call will confirm/deny the contradiction.

See FRONTIER_VISION_REPORT §2 — *Reconsolidation*.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from datetime import UTC, datetime

from core.graph.model import Node
from core.graph.storage import GraphStorage

logger = logging.getLogger(__name__)

# Similarity band for potential contradictions
CONTRA_SIM_LOW = 0.5
CONTRA_SIM_HIGH = 0.75


@dataclass(slots=True)
class ContraEvidence:
    """Evidence that a new text may contradict an existing belief."""

    belief_id: str
    belief_text: str
    new_text: str
    similarity: float
    detected_at: str


class ReconsolidationEngine:
    """Detect and manage belief contradictions.

    Parameters
    ----------
    storage:
        The :class:`~core.graph.storage.GraphStorage` instance.
    """

    def __init__(self, storage: GraphStorage) -> None:
        self.storage = storage

    async def check_contradiction(
        self,
        user_id: str,
        new_text: str,
        new_embedding: list[float] | None = None,
    ) -> list[ContraEvidence]:
        """Check whether *new_text* contradicts any existing BELIEF nodes.

        Returns a list of :class:`ContraEvidence` objects for beliefs whose
        embedding similarity falls in the contradiction band ``[0.5, 0.75]``.
        """
        if new_embedding is None:
            return []

        beliefs = await self.storage.find_nodes(user_id, node_type="BELIEF", limit=200)
        results: list[ContraEvidence] = []
        for belief in beliefs:
            if belief.embedding is None:
                continue
            sim = _cosine_similarity(new_embedding, belief.embedding)
            if CONTRA_SIM_LOW <= sim <= CONTRA_SIM_HIGH:
                results.append(
                    ContraEvidence(
                        belief_id=belief.id,
                        belief_text=belief.text or belief.name or "",
                        new_text=new_text,
                        similarity=round(sim, 4),
                        detected_at=datetime.now(UTC).isoformat(),
                    )
                )
                logger.info(
                    "Potential contradiction for belief %s (sim=%.3f)",
                    belief.id, sim,
                )
        return results

    async def update_belief(
        self,
        user_id: str,
        belief_id: str,
        evidence: ContraEvidence,
    ) -> Node:
        """Revise an existing BELIEF node with contra-evidence.

        Increments ``metadata.revision_count`` and appends to
        ``metadata.revision_history``.
        """
        node = await self.storage.get_node(belief_id)
        meta = dict(node.metadata)

        revision_count = int(meta.get("revision_count", 0)) + 1
        history = list(meta.get("revision_history", []))
        history.append({
            "text": evidence.new_text,
            "timestamp": evidence.detected_at,
            "reason": f"contradiction (sim={evidence.similarity})",
        })

        meta["revision_count"] = revision_count
        meta["revision_history"] = history
        meta["salience_score"] = 1.0  # refresh salience on revision

        revised = Node(
            id=node.id,
            user_id=node.user_id,
            type=node.type,
            name=node.name,
            text=f"{node.text or ''}\n[Revised] {evidence.new_text}",
            subtype=node.subtype,
            key=node.key,
            metadata=meta,
            created_at=node.created_at,
            embedding=node.embedding,
        )
        saved = await self.storage.upsert_node(revised)
        logger.info(
            "Revised belief %s (revision #%d) for user %s",
            belief_id, revision_count, user_id,
        )
        return saved


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    if len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)
