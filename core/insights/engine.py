"""InsightEngine — runs rules against graph data, persists INSIGHT nodes.

Lifecycle:
    1. Called after every background analysis (ORIENT + DECIDE).
    2. Loads all nodes/edges for the user (cheap SQLite reads).
    3. Runs every registered ``InsightRule`` against new + historical data.
    4. De-duplicates against already stored INSIGHTs (by pattern_type + title).
    5. Persists new INSIGHT nodes + GENERATES_INSIGHT edges.

The engine is intentionally light — rules do the heavy lifting.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from core.graph.api import GraphAPI
from core.graph.model import Edge, Node
from core.insights.rules import DEFAULT_RULES, InsightCandidate, InsightRule

logger = logging.getLogger(__name__)


class InsightEngine:
    """Detect and persist cross-pattern insights after every analysis."""

    # Insights with confidence below this are discarded
    MIN_CONFIDENCE = 0.4

    # Maximum insights to create per single analysis pass
    MAX_PER_PASS = 3

    def __init__(
        self,
        graph_api: GraphAPI,
        rules: list[InsightRule] | None = None,
    ) -> None:
        self.graph_api = graph_api
        self.rules = rules if rules is not None else list(DEFAULT_RULES)

    async def run(
        self,
        user_id: str,
        new_nodes: list[Node],
        new_edges: list[Edge],
        graph_context: dict,
    ) -> list[Node]:
        """Evaluate all rules and persist resulting INSIGHT nodes.

        Returns the list of newly created INSIGHT nodes (may be empty).
        """
        if not new_nodes and not new_edges:
            return []

        # Load full user graph for cross-referencing
        all_nodes = await self.graph_api.storage.find_nodes(user_id, limit=2000)
        all_edges = await self.graph_api.storage.list_edges(user_id)

        # Existing insights (for dedup)
        existing_insights = [
            n for n in all_nodes if n.type == "INSIGHT"
        ]
        existing_keys = {
            (n.metadata.get("pattern_type", ""), n.name or "")
            for n in existing_insights
        }

        # Run all rules
        candidates: list[InsightCandidate] = []
        for rule in self.rules:
            try:
                result = await rule.evaluate(
                    user_id=user_id,
                    new_nodes=new_nodes,
                    new_edges=new_edges,
                    all_nodes=all_nodes,
                    all_edges=all_edges,
                    graph_context=graph_context,
                )
                candidates.extend(result)
            except Exception as exc:
                logger.warning("InsightRule %s failed: %s", rule.name, exc)

        # Filter by confidence
        candidates = [c for c in candidates if c.confidence >= self.MIN_CONFIDENCE]

        # De-duplicate against existing
        fresh: list[InsightCandidate] = []
        for c in candidates:
            if (c.pattern_type, c.title) not in existing_keys:
                fresh.append(c)
                existing_keys.add((c.pattern_type, c.title))

        # Sort by confidence desc, take top N
        fresh.sort(key=lambda c: c.confidence, reverse=True)
        fresh = fresh[: self.MAX_PER_PASS]

        if not fresh:
            return []

        # Persist INSIGHT nodes
        created_nodes: list[Node] = []
        for candidate in fresh:
            key = f"insight:{candidate.pattern_type}:{candidate.title[:40].lower()}"
            node = Node(
                user_id=user_id,
                type="INSIGHT",
                name=candidate.title,
                text=candidate.description,
                key=key,
                metadata={
                    "pattern_type": candidate.pattern_type,
                    "confidence": candidate.confidence,
                    "severity": candidate.severity,
                    "related_node_ids": candidate.related_node_ids,
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    **candidate.metadata,
                },
            )
            saved = await self.graph_api.storage.upsert_node(node)
            created_nodes.append(saved)

            # Create GENERATES_INSIGHT edges from related nodes
            person = await self.graph_api.ensure_person_node(user_id)
            await self.graph_api.create_edge(
                user_id=user_id,
                source_node_id=person.id,
                target_node_id=saved.id,
                relation="GENERATES_INSIGHT",
            )
            for related_id in candidate.related_node_ids[:5]:
                await self.graph_api.create_edge(
                    user_id=user_id,
                    source_node_id=related_id,
                    target_node_id=saved.id,
                    relation="GENERATES_INSIGHT",
                )

            logger.info(
                "Insight created: [%s] %s (conf=%.2f sev=%s)",
                candidate.pattern_type,
                candidate.title,
                candidate.confidence,
                candidate.severity,
            )

        return created_nodes
