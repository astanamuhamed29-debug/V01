"""Hybrid vector search combining dense (cosine) and sparse (TF-IDF/BM25) scoring.

Provides two fusion strategies:

* **Weighted sum**: ``final = alpha * dense_score + (1 - alpha) * sparse_score``
* **Reciprocal Rank Fusion (RRF)**: ``score(d) = Σ 1 / (k + rank(d))``

All computation uses the Python standard library only (``math``, ``re``,
``collections``), with no third-party dependencies.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from core.graph.model import Node

from core.graph.model import get_node_embedding

__all__ = ["HybridSearchEngine", "rrf_fusion"]


# ---------------------------------------------------------------------------
# Sparse scoring helpers
# ---------------------------------------------------------------------------


def _tokenize(text: str) -> list[str]:
    """Lower-case word tokeniser that strips punctuation."""
    return re.findall(r"[а-яёa-z0-9]+", text.lower())


def _tf(tokens: list[str]) -> dict[str, float]:
    """Compute raw term frequency for *tokens*."""
    counts = Counter(tokens)
    total = len(tokens) or 1
    return {term: count / total for term, count in counts.items()}


def _idf(term: str, corpus: list[list[str]]) -> float:
    """Compute IDF for *term* over *corpus* (list of token-lists)."""
    df = sum(1 for doc in corpus if term in doc)
    if df == 0:
        return 0.0
    # Smoothed IDF using natural log (consistent with sklearn's TfidfVectorizer default)
    return math.log((len(corpus) + 1) / (df + 1)) + 1.0


def sparse_score(query_tokens: list[str], doc_tokens: list[str], corpus: list[list[str]]) -> float:
    """Return a TF-IDF-based sparse similarity score in [0, 1].

    Parameters
    ----------
    query_tokens:
        Tokenised query.
    doc_tokens:
        Tokenised document.
    corpus:
        All documents in the collection as token-lists (for IDF denominator).
    """
    if not query_tokens or not doc_tokens:
        return 0.0

    doc_tf = _tf(doc_tokens)
    score = 0.0
    for term in set(query_tokens):
        if term in doc_tf:
            idf_val = _idf(term, corpus)
            score += doc_tf[term] * idf_val

    # Normalise by query length so that longer queries don't dominate
    return score / len(set(query_tokens))


# ---------------------------------------------------------------------------
# RRF fusion
# ---------------------------------------------------------------------------


def rrf_fusion(
    dense_ranked: list[tuple[str, float]],
    sparse_ranked: list[tuple[str, float]],
    k: int = 60,
) -> list[tuple[str, float]]:
    """Reciprocal Rank Fusion of two ranked lists.

    Parameters
    ----------
    dense_ranked:
        ``[(node_id, dense_score), ...]`` sorted by score descending.
    sparse_ranked:
        ``[(node_id, sparse_score), ...]`` sorted by score descending.
    k:
        RRF hyper-parameter (default 60 per the original paper).

    Returns
    -------
    list of ``(node_id, rrf_score)`` sorted by RRF score descending.
    """
    scores: dict[str, float] = {}
    for rank, (node_id, _) in enumerate(dense_ranked, start=1):
        scores[node_id] = scores.get(node_id, 0.0) + 1.0 / (k + rank)
    for rank, (node_id, _) in enumerate(sparse_ranked, start=1):
        scores[node_id] = scores.get(node_id, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda item: item[1], reverse=True)


# ---------------------------------------------------------------------------
# Main engine
# ---------------------------------------------------------------------------


class HybridSearchEngine:
    """Combines dense (embedding cosine similarity) and sparse (TF-IDF) search.

    Parameters
    ----------
    alpha:
        Weight for dense scores in the weighted-sum fusion (0 = sparse only,
        1 = dense only). Default ``0.7``.
    """

    def __init__(self, alpha: float = 0.7) -> None:
        if not (0.0 <= alpha <= 1.0):
            raise ValueError(f"alpha must be in [0, 1], got {alpha}")
        self.alpha = alpha

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def search(
        self,
        query_text: str,
        query_embedding: list[float] | None,
        nodes: list["Node"],
        top_k: int = 10,
        use_rrf: bool = False,
    ) -> list[tuple["Node", float]]:
        """Return top-*k* nodes ranked by hybrid score.

        Parameters
        ----------
        query_text:
            Raw query string (used for sparse scoring).
        query_embedding:
            Dense query vector. If ``None``, only sparse scoring is used
            (``alpha`` is ignored and treated as ``0``).
        nodes:
            Candidate nodes to rank.
        top_k:
            Maximum number of results.
        use_rrf:
            If ``True``, use Reciprocal Rank Fusion instead of weighted sum.
        """
        if not nodes:
            return []

        query_tokens = _tokenize(query_text)
        corpus = [_tokenize((n.text or "") + " " + (n.name or "")) for n in nodes]

        sparse_scores: list[float] = [
            sparse_score(query_tokens, doc_tokens, corpus)
            for doc_tokens in corpus
        ]

        if use_rrf:
            return self._rrf_search(query_embedding, nodes, sparse_scores, top_k)

        if query_embedding is not None:
            dense_scores = [
                _cosine_similarity(query_embedding, emb) if emb else 0.0
                for n in nodes
                for emb in [get_node_embedding(n)]
            ]
            effective_alpha = self.alpha
        else:
            dense_scores = [0.0] * len(nodes)
            effective_alpha = 0.0

        combined = [
            (node, effective_alpha * d + (1.0 - effective_alpha) * s)
            for node, d, s in zip(nodes, dense_scores, sparse_scores)
        ]
        combined.sort(key=lambda item: item[1], reverse=True)
        return combined[:top_k]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _rrf_search(
        self,
        query_embedding: list[float] | None,
        nodes: list["Node"],
        sparse_scores: list[float],
        top_k: int,
    ) -> list[tuple["Node", float]]:
        """Perform RRF-based fusion and return top-*k* results."""
        node_by_id = {n.id: n for n in nodes}

        sparse_ranked = sorted(
            zip([n.id for n in nodes], sparse_scores),
            key=lambda item: item[1],
            reverse=True,
        )

        if query_embedding is not None:
            dense_pairs = [
                (n.id, _cosine_similarity(query_embedding, emb) if emb else 0.0)
                for n in nodes
                for emb in [get_node_embedding(n)]
            ]
            dense_ranked = sorted(dense_pairs, key=lambda item: item[1], reverse=True)
        else:
            dense_ranked = sparse_ranked  # treat as same list when no embedding

        fused = rrf_fusion(dense_ranked, sparse_ranked)
        results: list[tuple["Node", float]] = []
        for node_id, score in fused[:top_k]:
            if node_id in node_by_id:
                results.append((node_by_id[node_id], score))
        return results


# Canonical implementation in core.utils.math
from core.utils.math import cosine_similarity as _cosine_similarity  # noqa: E402
