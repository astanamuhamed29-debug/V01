from core.graph.api import GraphAPI
from core.graph.storage import GraphStorage


def test_graph_api_creates_nodes_and_edges(tmp_path):
    db_path = tmp_path / "test.db"
    api = GraphAPI(GraphStorage(db_path=db_path))

    person = api.ensure_person_node("me")
    project = api.find_or_create_node(
        user_id="me",
        node_type="PROJECT",
        key="project:self-os",
        name="SELF-OS",
    )
    task = api.create_node(user_id="me", node_type="TASK", text="набросать архитектуру", key="task:набросать архитектуру")
    api.create_edge(user_id="me", source_node_id=project.id, target_node_id=task.id, relation="HAS_TASK")
    api.create_edge(user_id="me", source_node_id=person.id, target_node_id=project.id, relation="OWNS_PROJECT")

    projects = api.get_user_nodes_by_type("me", "PROJECT")
    tasks = api.get_user_nodes_by_type("me", "TASK")
    subgraph = api.get_subgraph("me", ["PERSON", "PROJECT", "TASK"])

    assert len(projects) == 1
    assert projects[0].name == "SELF-OS"
    assert len(tasks) == 1
    assert len(subgraph["edges"]) >= 2
