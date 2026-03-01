"""Shared math utilities for SELF-OS.

Single-source ``cosine_similarity`` used across search, memory,
analytics, and storage modules. **No third-party dependencies.**
"""

from __future__ import annotations

import math

__all__ = ["cosine_similarity", "mean_embedding"]


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Return cosine similarity between two equal-length vectors.

    Returns 0.0 for zero-norm or mismatched-length inputs.
    """
    if len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def mean_embedding(embeddings: list[list[float]]) -> list[float] | None:
    """Average a list of equal-length embedding vectors.

    Returns ``None`` when *embeddings* is empty.
    """
    if not embeddings:
        return None
    dim = len(embeddings[0])
    mean = [0.0] * dim
    for emb in embeddings:
        for i, v in enumerate(emb):
            mean[i] += v
    n = len(embeddings)
    return [v / n for v in mean]
