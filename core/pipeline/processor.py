from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

from agents.reply_minimal import generate_reply
from core.graph.api import GraphAPI
from core.graph.model import Edge, Node
from core.journal.storage import JournalStorage
from core.pipeline import extractor_emotion, extractor_parts, extractor_semantic, router
from core.pipeline.events import EventBus


@dataclass(slots=True)
class ProcessResult:
    intent: str
    reply_text: str
    nodes: list[Node]
    edges: list[Edge]


class MessageProcessor:
    def __init__(
        self,
        graph_api: GraphAPI,
        journal: JournalStorage,
        event_bus: EventBus | None = None,
    ) -> None:
        self.graph_api = graph_api
        self.journal = journal
        self.event_bus = event_bus or EventBus()

    def process_message(
        self,
        user_id: str,
        text: str,
        *,
        source: str = "cli",
        timestamp: str | None = None,
    ) -> ProcessResult:
        ts = timestamp or datetime.now(timezone.utc).isoformat()

        self.journal.append(user_id=user_id, timestamp=ts, text=text, source=source)
        self.event_bus.publish("journal.appended", {"user_id": user_id, "text": text})

        intent = router.classify(text)

        person = self.graph_api.ensure_person_node(user_id)

        semantic_nodes, semantic_edges = extractor_semantic.extract(user_id, text, intent, person.id)
        parts_nodes, parts_edges = extractor_parts.extract(user_id, text, intent, person.id)
        emotion_nodes, emotion_edges = extractor_emotion.extract(user_id, text, intent, person.id)

        nodes = [*semantic_nodes, *parts_nodes, *emotion_nodes]
        edges = [*semantic_edges, *parts_edges, *emotion_edges]

        created_nodes, created_edges = self.graph_api.apply_changes(user_id, nodes, edges)

        projects = self.graph_api.get_user_nodes_by_type(user_id, "PROJECT")
        tasks = [node for node in created_nodes if node.type == "TASK"]
        if tasks and projects:
            for task in tasks:
                self.graph_api.create_edge(
                    user_id=user_id,
                    source_node_id=projects[0].id,
                    target_node_id=task.id,
                    relation="HAS_TASK",
                )

        reply_text = generate_reply(
            text=text,
            intent=intent,
            extracted_structures={
                "nodes": [*created_nodes],
                "edges": [*created_edges],
            },
        )

        self.event_bus.publish(
            "pipeline.processed",
            {
                "user_id": user_id,
                "intent": intent,
                "nodes": len(created_nodes),
                "edges": len(created_edges),
            },
        )

        return ProcessResult(intent=intent, reply_text=reply_text, nodes=created_nodes, edges=created_edges)

    def process(
        self,
        user_id: str,
        text: str,
        *,
        source: str = "cli",
        timestamp: str | None = None,
    ) -> ProcessResult:
        return self.process_message(
            user_id=user_id,
            text=text,
            source=source,
            timestamp=timestamp,
        )
