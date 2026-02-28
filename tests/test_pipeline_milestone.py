from core.graph.api import GraphAPI
from core.graph.storage import GraphStorage
from core.journal.storage import JournalStorage
from core.pipeline.processor import MessageProcessor


def test_milestone_scenario_builds_expected_graph(tmp_path):
    db_path = tmp_path / "self_os.db"
    api = GraphAPI(GraphStorage(db_path=db_path))
    journal = JournalStorage(db_path=db_path)
    processor = MessageProcessor(graph_api=api, journal=journal)

    messages = [
        "Хочу сделать свою личную ОС SELF-OS.",
        "Надо выделить вечер, чтобы набросать архитектуру.",
        "Я боюсь, что не вывезу такой большой проект.",
    ]

    for text in messages:
        processor.process_message(user_id="me", text=text, source="cli")

    notes = api.get_user_nodes_by_type("me", "NOTE")
    projects = api.get_user_nodes_by_type("me", "PROJECT")
    tasks = api.get_user_nodes_by_type("me", "TASK")
    beliefs = api.get_user_nodes_by_type("me", "BELIEF")

    assert len(notes) == 3
    assert len(projects) == 1
    assert projects[0].name == "SELF-OS"
    assert len(tasks) >= 1
    assert any("набросать архитектуру" in (task.text or "") for task in tasks)
    assert len(beliefs) == 1

    edges = api.storage.list_edges("me")
    relations = {edge.relation for edge in edges}
    assert "OWNS_PROJECT" in relations
    assert "HAS_TASK" in relations
    assert "HOLDS_BELIEF" in relations


def test_relocation_phrase_creates_relocation_project(tmp_path):
    db_path = tmp_path / "self_os.db"
    api = GraphAPI(GraphStorage(db_path=db_path))
    journal = JournalStorage(db_path=db_path)
    processor = MessageProcessor(graph_api=api, journal=journal)

    processor.process(user_id="321", text="Привет, я хочу переехать", source="telegram")

    projects = api.get_user_nodes_by_type("321", "PROJECT")
    assert any((project.name or "").lower() == "переезд" for project in projects)
