"""Tests for core/search/hybrid_search.py."""

import pytest

from core.graph.model import Node
from core.search.hybrid_search import HybridSearchEngine, rrf_fusion, sparse_score, _tokenize


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _node(node_id: str, text: str, embedding: list[float] | None = None) -> Node:
    return Node(
        id=node_id,
        user_id="u1",
        type="NOTE",
        text=text,
        metadata={"embedding": embedding} if embedding is not None else {},
    )


# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------


def test_tokenize_basic():
    tokens = _tokenize("Hello World 123")
    assert "hello" in tokens
    assert "world" in tokens
    assert "123" in tokens


def test_tokenize_russian():
    tokens = _tokenize("Я хочу создать проект")
    assert "я" in tokens
    assert "проект" in tokens


# ---------------------------------------------------------------------------
# Sparse score
# ---------------------------------------------------------------------------


def test_sparse_score_exact_match():
    q = _tokenize("проект задача")
    d = _tokenize("проект задача важная")
    corpus = [d]
    score = sparse_score(q, d, corpus)
    assert score > 0.0


def test_sparse_score_no_match():
    q = _tokenize("кот")
    d = _tokenize("собака мяч")
    corpus = [d]
    score = sparse_score(q, d, corpus)
    assert score == 0.0


def test_sparse_score_empty_query():
    score = sparse_score([], ["word"], [["word"]])
    assert score == 0.0


# ---------------------------------------------------------------------------
# RRF fusion
# ---------------------------------------------------------------------------


def test_rrf_fusion_merges_lists():
    dense = [("a", 0.9), ("b", 0.7), ("c", 0.5)]
    sparse = [("b", 0.8), ("a", 0.6), ("d", 0.4)]
    result = rrf_fusion(dense, sparse)
    ids = [r[0] for r in result]
    # a and b should be high ranked
    assert "a" in ids[:2]
    assert "b" in ids[:2]


def test_rrf_fusion_scores_positive():
    dense = [("x", 1.0)]
    sparse = [("x", 1.0)]
    result = rrf_fusion(dense, sparse)
    assert result[0][1] > 0


# ---------------------------------------------------------------------------
# HybridSearchEngine — weighted sum
# ---------------------------------------------------------------------------


def test_hybrid_search_returns_top_k():
    engine = HybridSearchEngine(alpha=0.5)
    nodes = [
        _node("n1", "проект задача разработка"),
        _node("n2", "кот собака мяч"),
        _node("n3", "проект планирование"),
    ]
    results = engine.search("проект", None, nodes, top_k=2)
    assert len(results) == 2
    assert all(isinstance(n, Node) for n, _ in results)


def test_hybrid_search_sparse_only_no_embedding():
    engine = HybridSearchEngine(alpha=0.9)
    nodes = [
        _node("n1", "задача важная"),
        _node("n2", "совсем другое"),
    ]
    results = engine.search("задача", None, nodes, top_k=2)
    ids = [n.id for n, _ in results]
    assert ids[0] == "n1"


def test_hybrid_search_with_embedding():
    engine = HybridSearchEngine(alpha=0.7)
    nodes = [
        _node("n1", "проект", embedding=[1.0, 0.0]),
        _node("n2", "кот", embedding=[0.0, 1.0]),
    ]
    results = engine.search("проект", [1.0, 0.0], nodes, top_k=2)
    assert results[0][0].id == "n1"


def test_hybrid_search_rrf():
    engine = HybridSearchEngine(alpha=0.7)
    nodes = [
        _node("n1", "проект задача"),
        _node("n2", "кот"),
    ]
    results = engine.search("проект", None, nodes, top_k=2, use_rrf=True)
    assert len(results) >= 1


def test_hybrid_search_empty_nodes():
    engine = HybridSearchEngine()
    assert engine.search("query", None, []) == []


def test_hybrid_search_invalid_alpha():
    with pytest.raises(ValueError):
        HybridSearchEngine(alpha=1.5)
