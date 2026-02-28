"""Graph analytics — simplified PageRank-like node importance metric.

:func:`compute_node_importance` runs a lightweight iterative PageRank
computation directly on the user's SELF-Graph without requiring any
third-party graph library.  The result can be used to rank nodes when
building LLM context, ensuring the most "central" nodes receive more
attention.

Algorithm
---------
Standard PageRank with damping factor *d = 0.85* and a configurable number
of iterations.  Edge weights from :func:`~core.graph.model.edge_weight` are
optionally applied so that recent connections carry more influence.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from core.graph.storage import GraphStorage

__all__ = ["compute_node_importance"]

_DAMPING = 0.85      # Standard PageRank damping factor
_ITERATIONS = 20    # Number of power-iteration steps (convergence is fast for sparse graphs)
_MIN_SCORE = 1e-9  # Floor to avoid zero scores on isolated nodes


async def compute_node_importance(
    user_id: str,
    storage: "GraphStorage",
    damping: float = _DAMPING,
    iterations: int = _ITERATIONS,
    use_temporal_weights: bool = True,
) -> dict[str, float]:
    """Compute a simplified PageRank score for every node owned by *user_id*.

    Parameters
    ----------
    user_id:
        Graph owner.
    storage:
        :class:`~core.graph.storage.GraphStorage` instance.
    damping:
        PageRank damping factor (default 0.85).
    iterations:
        Number of power-iteration steps (default 20).
    use_temporal_weights:
        When ``True``, edge weights are modulated by
        :func:`~core.graph.model.edge_weight` (temporal decay).

    Returns
    -------
    dict[str, float]
        Mapping ``node_id → PageRank score``.  Scores are normalised so
        they sum to 1.0.  Returns an empty dict if the user has no nodes.
    """
    from core.graph.model import edge_weight as _edge_weight

    nodes = await storage.find_nodes(user_id, limit=2000)
    if not nodes:
        return {}

    edges = await storage.list_edges(user_id)
    node_ids = [n.id for n in nodes]
    n = len(node_ids)
    idx: dict[str, int] = {nid: i for i, nid in enumerate(node_ids)}

    # Build weighted adjacency: in_weights[i] = list of (source_idx, weight)
    in_weights: list[list[tuple[int, float]]] = [[] for _ in range(n)]
    for edge in edges:
        src_idx = idx.get(edge.source_node_id)
        tgt_idx = idx.get(edge.target_node_id)
        if src_idx is None or tgt_idx is None:
            continue
        w = _edge_weight(edge) if use_temporal_weights else 1.0
        in_weights[tgt_idx].append((src_idx, w))

    # Out-degree weighted sum per node
    out_weight_sum: list[float] = [0.0] * n
    for tgt_idx in range(n):
        for src_idx, w in in_weights[tgt_idx]:
            out_weight_sum[src_idx] += w

    # Initialise PageRank uniformly
    rank = [1.0 / n] * n

    for _ in range(iterations):
        new_rank = [(1.0 - damping) / n] * n
        for tgt_idx in range(n):
            for src_idx, w in in_weights[tgt_idx]:
                out_total = out_weight_sum[src_idx]
                if out_total > 0:
                    new_rank[tgt_idx] += damping * rank[src_idx] * (w / out_total)
                else:
                    # Dangling node — distribute evenly
                    new_rank[tgt_idx] += damping * rank[src_idx] / n
        rank = new_rank

    # Normalise
    total = sum(rank) or 1.0
    return {node_ids[i]: max(_MIN_SCORE, rank[i] / total) for i in range(n)}
