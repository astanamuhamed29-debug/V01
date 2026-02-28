from datetime import datetime, timedelta, timezone

from core.graph.model import Edge, edge_weight


def test_edge_weight_fresh():
    edge = Edge(
        user_id="u1",
        source_node_id="a",
        target_node_id="b",
        relation="RELATES_TO",
        created_at=datetime.now(timezone.utc).isoformat(),
    )
    value = edge_weight(edge)
    assert 0.98 <= value <= 1.0


def test_edge_weight_halflife():
    edge = Edge(
        user_id="u1",
        source_node_id="a",
        target_node_id="b",
        relation="RELATES_TO",
        created_at=(datetime.now(timezone.utc) - timedelta(days=30)).isoformat(),
    )
    value = edge_weight(edge, half_life_days=30.0)
    assert 0.45 <= value <= 0.55


def test_edge_weight_old():
    edge = Edge(
        user_id="u1",
        source_node_id="a",
        target_node_id="b",
        relation="RELATES_TO",
        created_at=(datetime.now(timezone.utc) - timedelta(days=120)).isoformat(),
    )
    value = edge_weight(edge, half_life_days=30.0)
    assert value < 0.1
