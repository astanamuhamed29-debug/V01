import logging

from core.llm.parser import is_minimal_payload, map_payload_to_graph, parse_json_payload


def test_is_minimal_empty():
    assert is_minimal_payload({}) is True


def test_is_minimal_reflection():
    assert is_minimal_payload({"intent": "REFLECTION"}) is True


def test_not_minimal_with_nodes():
    assert is_minimal_payload({"intent": "META", "nodes": [{"type": "VALUE"}]}) is False


def test_parse_fenced_json():
    raw = '```json\n{"intent": "META"}\n```'
    assert parse_json_payload(raw) == {"intent": "META"}


def test_parse_think_tag():
    raw = '<think>внутренние рассуждения</think>{"intent": "META"}'
    assert parse_json_payload(raw) == {"intent": "META"}


def test_map_filters_unknown_type():
    data = {"nodes": [{"type": "ALIEN", "id": "x"}], "edges": []}
    nodes, edges = map_payload_to_graph(user_id="u1", person_id="p1", data=data)
    assert nodes == []
    assert edges == []


def test_parser_with_reasoning_block(caplog):
    raw = (
        '{"_reasoning":{"situation":"конфликт на работе","appraisal":"меня не слышат",'
        '"affect":"злость","defenses":"критик","core_needs":"уважение"},'
        '"intent":"FEELING_REPORT","nodes":[],"edges":[]}'
    )
    with caplog.at_level(logging.DEBUG):
        data = parse_json_payload(raw)

    assert data["intent"] == "FEELING_REPORT"
    assert "_reasoning" in data
    assert "Extractor reasoning block" in caplog.text


def test_map_generates_soma_key_when_missing():
    data = {
        "nodes": [
            {"id": "s1", "type": "SOMA", "metadata": {"location": "грудь", "sensation": "тяжесть"}},
        ],
        "edges": [],
    }
    nodes, _edges = map_payload_to_graph(user_id="u1", person_id="p1", data=data)
    assert len(nodes) == 1
    assert nodes[0].type == "SOMA"
    assert nodes[0].key is not None
    assert str(nodes[0].key).startswith("soma:")
