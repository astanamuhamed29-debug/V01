"""MemoryConsolidator — Stage 2 memory lifecycle management.

Implements the three-phase memory lifecycle from FRONTIER_VISION_REPORT §2:

* **consolidate()** — cluster similar low-retention NOTE nodes into BELIEF/THOUGHT
  nodes using embedding similarity (runs daily).
* **abstract()** — summarise BELIEF/THOUGHT nodes into higher-level semantic nodes
  (runs weekly). *Placeholder — requires LLM integration.*
* **forget()** — soft-delete edges and orphan nodes that fall below the forgetting
  threshold (runs weekly).

All operations are **idempotent** and safe to re-run.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from datetime import UTC, datetime
from uuid import uuid4

from core.graph.model import Node, ebbinghaus_retention, ensure_metadata_defaults
from core.graph.storage import GraphStorage

logger = logging.getLogger(__name__)

# ── Configuration knobs ──────────────────────────────────────────
CONSOLIDATION_RETENTION_THRESHOLD = 0.3
CONSOLIDATION_SIMILARITY_THRESHOLD = 0.82
CONSOLIDATION_MIN_CLUSTER_SIZE = 2
FORGETTING_EDGE_THRESHOLD = 0.05
FORGETTING_NODE_THRESHOLD = 0.1
PROTECTED_TYPES = frozenset({"BELIEF", "NEED", "VALUE"})
PROTECTED_REVIEW_MIN = 2
MAX_ARCHETYPE_NAME_LENGTH = 120


@dataclass(slots=True)
class ConsolidationReport:
    """Summary returned by :meth:`MemoryConsolidator.consolidate`."""

    clusters_found: int
    nodes_merged: int
    new_nodes_created: int


@dataclass(slots=True)
class AbstractionReport:
    """Summary returned by :meth:`MemoryConsolidator.abstract`."""

    candidates: int
    abstracted: int


@dataclass(slots=True)
class ForgetReport:
    """Summary returned by :meth:`MemoryConsolidator.forget`."""

    edges_removed: int
    nodes_tombstoned: int


class MemoryConsolidator:
    """Manages the full memory lifecycle for a user's knowledge graph.

    Parameters
    ----------
    storage:
        The :class:`~core.graph.storage.GraphStorage` instance.
    llm_client:
        Optional LLM client for abstraction summarisation.  When
        ``None``, :meth:`abstract` falls back to counting candidates
        without actually merging them.
    """

    def __init__(
        self,
        storage: GraphStorage,
        llm_client: object | None = None,
    ) -> None:
        self.storage = storage
        self._llm_client = llm_client

    # ── 1. Consolidate ────────────────────────────────────────────

    async def consolidate(
        self,
        user_id: str,
        *,
        retention_threshold: float = CONSOLIDATION_RETENTION_THRESHOLD,
        similarity_threshold: float = CONSOLIDATION_SIMILARITY_THRESHOLD,
        min_cluster_size: int = CONSOLIDATION_MIN_CLUSTER_SIZE,
    ) -> ConsolidationReport:
        """Cluster low-retention NOTE nodes by embedding similarity and merge.

        Algorithm (from FRONTIER_VISION_REPORT §2):
        1. Find NOTE nodes with salience_score ≤ *retention_threshold*.
        2. Cluster by cosine similarity ≥ *similarity_threshold*.
        3. For each cluster create one BELIEF or THOUGHT node.
        4. Re-point edges via ``storage.merge_nodes()``.
        """
        candidates = await self.storage.get_nodes_by_retention(
            user_id,
            max_retention=retention_threshold,
            node_types=["NOTE"],
            limit=500,
        )

        # Only keep candidates that have embeddings
        embedded = [n for n in candidates if n.embedding is not None]
        if len(embedded) < min_cluster_size:
            return ConsolidationReport(clusters_found=0, nodes_merged=0, new_nodes_created=0)

        clusters = _cluster_by_embedding(embedded, similarity_threshold, min_cluster_size)
        total_merged = 0
        total_created = 0

        for cluster in clusters:
            # Combine texts
            combined_text = "\n---\n".join(n.text or n.name or "" for n in cluster)
            source_ids = [n.id for n in cluster]

            merged_node = Node(
                id=str(uuid4()),
                user_id=user_id,
                type="BELIEF",
                name=f"consolidated ({len(cluster)} notes)",
                text=combined_text[:2000],
                key=f"consolidated:{uuid4().hex[:8]}",
                metadata=ensure_metadata_defaults({
                    "abstraction_level": 1,
                    "consolidation_source": source_ids,
                    "salience_score": 1.0,
                }),
                created_at=datetime.now(UTC).isoformat(),
                embedding=_mean_embedding([n.embedding for n in cluster if n.embedding]),
            )

            await self.storage.merge_nodes(user_id, source_ids, merged_node)
            total_merged += len(cluster)
            total_created += 1
            logger.info(
                "Consolidated %d NOTE nodes into BELIEF %s for user %s",
                len(cluster), merged_node.id, user_id,
            )

        return ConsolidationReport(
            clusters_found=len(clusters),
            nodes_merged=total_merged,
            new_nodes_created=total_created,
        )

    # ── 2. Abstract ───────────────────────────────────────────────

    async def abstract(self, user_id: str) -> AbstractionReport:
        """Promote BELIEF/THOUGHT nodes to higher abstraction level.

        When an LLM client is available, clusters of BELIEF nodes at
        ``abstraction_level == 1`` are summarised into a single archetype
        BELIEF at ``abstraction_level == 2`` via LLM.

        Without an LLM client the method counts candidates only (backward
        compatible with the Sprint-0 placeholder behaviour).
        """
        nodes = await self.storage.find_nodes(user_id, node_type="BELIEF", limit=500)
        candidates = [
            n for n in nodes
            if n.metadata.get("abstraction_level", 0) == 1
        ]

        if not candidates or self._llm_client is None:
            return AbstractionReport(candidates=len(candidates), abstracted=0)

        # Cluster candidates by embedding similarity (reuse consolidation clustering)
        embedded = [n for n in candidates if n.embedding is not None]
        clusters = _cluster_by_embedding(
            embedded,
            threshold=CONSOLIDATION_SIMILARITY_THRESHOLD,
            min_size=CONSOLIDATION_MIN_CLUSTER_SIZE,
        )

        abstracted = 0
        for cluster in clusters:
            texts = [n.text or n.name or "" for n in cluster]
            summary = await _llm_summarise(self._llm_client, texts)
            if not summary:
                continue

            source_ids = [n.id for n in cluster]
            archetype = Node(
                id=str(uuid4()),
                user_id=user_id,
                type="BELIEF",
                name=summary[:MAX_ARCHETYPE_NAME_LENGTH],
                text=summary,
                key=f"archetype:{uuid4().hex[:8]}",
                metadata=ensure_metadata_defaults({
                    "abstraction_level": 2,
                    "consolidation_source": source_ids,
                    "salience_score": 1.0,
                }),
                created_at=datetime.now(UTC).isoformat(),
                embedding=_mean_embedding(
                    [n.embedding for n in cluster if n.embedding]
                ),
            )

            await self.storage.merge_nodes(user_id, source_ids, archetype)
            abstracted += 1
            logger.info(
                "Abstracted %d beliefs into archetype %s for user %s",
                len(cluster),
                archetype.id,
                user_id,
            )

        return AbstractionReport(candidates=len(candidates), abstracted=abstracted)

    # ── 3. Forget ─────────────────────────────────────────────────

    async def forget(
        self,
        user_id: str,
        *,
        edge_threshold: float = FORGETTING_EDGE_THRESHOLD,
        node_threshold: float = FORGETTING_NODE_THRESHOLD,
    ) -> ForgetReport:
        """Remove stale edges and tombstone orphan nodes.

        Rules (from FRONTIER_VISION_REPORT §2):
        * Edges with ``ebbinghaus_retention < edge_threshold`` are deleted.
        * Nodes with no remaining edges and salience < *node_threshold*
          are soft-deleted (**tombstoned**).
        * BELIEF / NEED / VALUE nodes with ``review_count ≥ 2`` are never deleted.
        """
        await self.storage._ensure_initialized()
        conn = await self.storage._get_conn()

        # --- edges ---
        edges = await self.storage.list_edges(user_id)
        edges_removed = 0
        for edge in edges:
            review_count = int(edge.metadata.get("review_count", 0))
            retention = ebbinghaus_retention(edge, review_count=review_count)
            if retention < edge_threshold:
                await conn.execute("DELETE FROM edges WHERE id = ?", (edge.id,))
                edges_removed += 1

        # --- orphan nodes ---
        nodes = await self.storage.find_nodes(user_id, limit=1000)
        remaining_edges = await self.storage.list_edges(user_id)
        connected_ids: set[str] = set()
        for e in remaining_edges:
            connected_ids.add(e.source_node_id)
            connected_ids.add(e.target_node_id)

        nodes_tombstoned = 0
        for node in nodes:
            if node.id in connected_ids:
                continue
            if node.type == "PERSON":
                continue
            if node.type in PROTECTED_TYPES and int(node.metadata.get("review_count", 0)) >= PROTECTED_REVIEW_MIN:
                continue
            salience = float(node.metadata.get("salience_score", 1.0))
            if salience < node_threshold:
                await self.storage.soft_delete_node(node.id)
                nodes_tombstoned += 1

        await conn.commit()
        logger.info(
            "Forget pass for user %s: %d edges removed, %d nodes tombstoned",
            user_id, edges_removed, nodes_tombstoned,
        )
        return ForgetReport(edges_removed=edges_removed, nodes_tombstoned=nodes_tombstoned)


# ── Helpers ───────────────────────────────────────────────────────


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    if len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def _mean_embedding(embeddings: list[list[float]]) -> list[float] | None:
    """Average a list of equal-length embedding vectors."""
    if not embeddings:
        return None
    dim = len(embeddings[0])
    mean = [0.0] * dim
    for emb in embeddings:
        for i, v in enumerate(emb):
            mean[i] += v
    n = len(embeddings)
    return [v / n for v in mean]


def _cluster_by_embedding(
    nodes: list[Node],
    threshold: float,
    min_size: int,
) -> list[list[Node]]:
    """Simple greedy single-linkage clustering by cosine similarity."""
    used: set[int] = set()
    clusters: list[list[Node]] = []
    for i, node_i in enumerate(nodes):
        if i in used:
            continue
        cluster = [node_i]
        used.add(i)
        for j in range(i + 1, len(nodes)):
            if j in used:
                continue
            node_j = nodes[j]
            if node_i.embedding and node_j.embedding:
                sim = _cosine_similarity(node_i.embedding, node_j.embedding)
                if sim >= threshold:
                    cluster.append(node_j)
                    used.add(j)
        if len(cluster) >= min_size:
            clusters.append(cluster)
    return clusters


_ABSTRACTION_PROMPT = (
    "You are a psychologist assistant. "
    "Summarise the following beliefs into ONE concise archetype belief "
    "(1-2 sentences, in the same language as the input). "
    "Return ONLY the summary text, nothing else.\n\n"
)


async def _llm_summarise(llm_client: object, texts: list[str]) -> str | None:
    """Use an LLM to summarise a cluster of belief texts into one archetype.

    Accepts any object that has an ``extract_all`` or ``generate_live_reply``
    async method (i.e. the existing :class:`~core.llm_client.LLMClient` protocol).
    Falls back gracefully on failure.
    """
    combined = "\n---\n".join(texts)
    prompt = _ABSTRACTION_PROMPT + combined

    # Try generate_live_reply first (produces free-form text)
    gen_fn = getattr(llm_client, "generate_live_reply", None)
    if gen_fn is not None:
        try:
            result = await gen_fn(
                user_text=prompt,
                intent="ABSTRACTION",
                mood_context=None,
                parts_context=None,
                graph_context=None,
            )
            if result and result.strip():
                return result.strip()
        except Exception as exc:
            logger.warning("LLM abstraction via generate_live_reply failed: %s", exc)

    # Fallback: use extract_all and parse text from result
    extract_fn = getattr(llm_client, "extract_all", None)
    if extract_fn is not None:
        try:
            result = await extract_fn(prompt, "ABSTRACTION")
            if isinstance(result, str) and result.strip():
                return result.strip()
            if isinstance(result, dict):
                return str(result.get("summary", result.get("text", "")))[:500] or None
        except Exception as exc:
            logger.warning("LLM abstraction via extract_all failed: %s", exc)

    return None
