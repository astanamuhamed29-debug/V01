import asyncio

from core.graph.api import GraphAPI
from core.graph.model import Edge, Node
from core.graph.storage import GraphStorage


def test_graph_api_creates_nodes_and_edges(tmp_path):
    async def scenario() -> None:
        db_path = tmp_path / "test.db"
        api = GraphAPI(GraphStorage(db_path=db_path))

        try:
            person = await api.ensure_person_node("me")
            project = await api.find_or_create_node(
                user_id="me",
                node_type="PROJECT",
                key="project:self-os",
                name="SELF-OS",
            )
            await api.create_node(user_id="me", node_type="TASK", text="набросать архитектуру", key="task:набросать архитектуру")
            task = (await api.get_user_nodes_by_type("me", "TASK"))[0]
            await api.create_edge(user_id="me", source_node_id=project.id, target_node_id=task.id, relation="HAS_TASK")
            await api.create_edge(user_id="me", source_node_id=person.id, target_node_id=project.id, relation="OWNS_PROJECT")

            projects = await api.get_user_nodes_by_type("me", "PROJECT")
            tasks = await api.get_user_nodes_by_type("me", "TASK")
            subgraph = await api.get_subgraph("me", ["PERSON", "PROJECT", "TASK"])

            assert len(projects) == 1
            assert projects[0].name == "SELF-OS"
            assert len(tasks) == 1
            assert len(subgraph["edges"]) >= 2
        finally:
            await api.storage.close()

    asyncio.run(scenario())


def test_apply_changes_dedups_nodes_by_key_and_edges(tmp_path):
    async def scenario() -> None:
        db_path = tmp_path / "test.db"
        api = GraphAPI(GraphStorage(db_path=db_path))
        try:
            person = await api.ensure_person_node("me")

            first_project = Node(
            id="tmp-project-1",
            user_id="me",
            type="PROJECT",
            name="SELF-OS",
            key="project:self-os",
            metadata={"source": "first"},
        )
            first_edge = Edge(
            user_id="me",
            source_node_id=person.id,
            target_node_id=first_project.id,
            relation="OWNS_PROJECT",
        )
            await api.apply_changes("me", [first_project], [first_edge])

            second_project = Node(
            id="tmp-project-2",
            user_id="me",
            type="PROJECT",
            name="SELF-OS",
            key="project:self-os",
            metadata={"source": "second", "updated": True},
        )
            second_edge = Edge(
            user_id="me",
            source_node_id=person.id,
            target_node_id=second_project.id,
            relation="OWNS_PROJECT",
        )
            await api.apply_changes("me", [second_project], [second_edge])

            projects = [node for node in await api.get_user_nodes_by_type("me", "PROJECT") if node.key == "project:self-os"]
            assert len(projects) == 1
            assert projects[0].metadata.get("source") == "second"
            assert projects[0].metadata.get("updated") is True

            subgraph = await api.get_subgraph("me", ["PERSON", "PROJECT"])
            own_edges = [
                edge
                for edge in subgraph["edges"]
                if edge.relation == "OWNS_PROJECT" and edge.source_node_id == person.id and edge.target_node_id == projects[0].id
            ]
            assert len(own_edges) == 1
        finally:
            await api.storage.close()

    asyncio.run(scenario())


def test_normalize_key_dedup(tmp_path):
    async def scenario() -> None:
        db_path = tmp_path / "test.db"
        api = GraphAPI(GraphStorage(db_path=db_path))

        try:
            node1 = Node(
            id="n-key-1",
            user_id="me",
            type="PROJECT",
            name="SELF-OS",
            key="project:SELF-OS",
        )
            node2 = Node(
            id="n-key-2",
            user_id="me",
            type="PROJECT",
            name="SELF-OS",
            key="project:self-os",
        )

            await api.apply_changes("me", [node1, node2], [])

            projects = [node for node in await api.get_user_nodes_by_type("me", "PROJECT") if node.key == "project:self-os"]
            assert len(projects) == 1
        finally:
            await api.storage.close()

    asyncio.run(scenario())


def test_edge_dedup(tmp_path):
    async def scenario() -> None:
        db_path = tmp_path / "test.db"
        api = GraphAPI(GraphStorage(db_path=db_path))
        try:
            person = await api.ensure_person_node("me")

            project = Node(
            id="proj-1",
            user_id="me",
            type="PROJECT",
            name="SELF-OS",
            key="project:self-os",
        )
            edge = Edge(
            user_id="me",
            source_node_id=person.id,
            target_node_id=project.id,
            relation="OWNS_PROJECT",
        )

            await api.apply_changes("me", [project], [edge])

            project_again = Node(
            id="proj-2",
            user_id="me",
            type="PROJECT",
            name="SELF-OS",
            key="project:self-os",
        )
            edge_again = Edge(
            user_id="me",
            source_node_id=person.id,
            target_node_id=project_again.id,
            relation="OWNS_PROJECT",
        )
            await api.apply_changes("me", [project_again], [edge_again])

            subgraph = await api.get_subgraph("me", ["PERSON", "PROJECT"])
            own_edges = [
                item
                for item in subgraph["edges"]
                if item.source_node_id == person.id and item.relation == "OWNS_PROJECT"
            ]
            assert len(own_edges) == 1
        finally:
            await api.storage.close()

    asyncio.run(scenario())
