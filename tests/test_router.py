from core.pipeline.router import classify


def test_router_classifies_task_like():
    assert classify("Надо сделать архитектуру") == "TASK_LIKE"


def test_router_classifies_feeling_report():
    assert classify("Я боюсь, что не вывезу") == "FEELING_REPORT"


def test_router_classifies_meta():
    assert classify("Какие есть команды?") == "META"


def test_meta_intent():
    assert classify("в чем польза") == "META"


def test_meta_intent_2():
    assert classify("зачем это всё") == "META"
