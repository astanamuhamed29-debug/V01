"""IdentityProfileBuilder — constructs an IdentityProfile from system memory.

The builder reads from the graph storage (if available) and synthesises a
structured :class:`~core.identity.schema.IdentityProfile`.  It degrades
gracefully: if no storage is provided or queries fail, it returns an empty
but valid profile rather than raising.

Typical usage::

    from core.graph.storage import GraphStorage
    from core.identity.builder import IdentityProfileBuilder

    storage = GraphStorage(db_path="memory.db")
    builder = IdentityProfileBuilder(graph_storage=storage)
    profile = await builder.build(user_id="user_123")
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import Any

from core.identity.schema import (
    DomainProfile,
    IdentityProfile,
    ProfileGap,
)

logger = logging.getLogger(__name__)

# Node types we query from the graph
_NODE_TYPES = ("PROJECT", "TASK", "BELIEF", "NEED", "INSIGHT", "VALUE")

# Domains inferred from project/task metadata when no explicit domain is set
_FALLBACK_DOMAIN = "general"

# Confidence weighting for domain facts accumulation
_FACT_CONFIDENCE_WEIGHT = 0.1
_BASE_DOMAIN_CONFIDENCE = 0.1


class IdentityProfileBuilder:
    """Builds an :class:`IdentityProfile` from existing system memory.

    Parameters
    ----------
    graph_storage:
        Optional :class:`~core.graph.storage.GraphStorage` instance.  When
        *None* the builder still returns a valid (empty) profile.
    graph_api:
        Alternative graph accessor (must expose ``find_nodes`` / ``get_nodes``
        compatible with GraphStorage).  Takes precedence over *graph_storage*
        when both are supplied.
    """

    def __init__(
        self,
        graph_storage: Any | None = None,
        graph_api: Any | None = None,
    ) -> None:
        self._storage = graph_api if graph_api is not None else graph_storage

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def build(self, user_id: str) -> IdentityProfile:
        """Return a populated :class:`IdentityProfile` for *user_id*.

        Always returns a valid object; errors are logged and result in an
        empty (zero-confidence) profile.
        """
        now = datetime.now(UTC).isoformat()
        profile = IdentityProfile(
            user_id=user_id,
            created_at=now,
            updated_at=now,
        )

        if self._storage is None:
            self._add_bootstrap_gaps(profile)
            return profile

        try:
            await self._populate_from_graph(profile)
        except Exception as exc:
            logger.warning(
                "IdentityProfileBuilder: graph query failed for user=%s: %s",
                user_id,
                exc,
            )
            self._add_bootstrap_gaps(profile)

        return profile

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _populate_from_graph(self, profile: IdentityProfile) -> None:
        """Query graph nodes and fill in the profile fields."""
        user_id = profile.user_id
        evidence: list[str] = []
        domain_data: dict[str, dict[str, Any]] = {}  # domain → aggregated info

        for node_type in ("PROJECT", "TASK", "BELIEF", "NEED", "INSIGHT", "VALUE"):
            nodes = await self._find_nodes(user_id, node_type)
            for node in nodes:
                node_id: str = getattr(node, "id", None) or node.get("id", "")
                if node_id:
                    evidence.append(node_id)
                self._ingest_node(profile, node, node_type, domain_data)

        # Build DomainProfile objects from accumulated data
        for domain, data in domain_data.items():
            dp = DomainProfile(
                domain=domain,
                summary=data.get("summary", ""),
                goals=data.get("goals", []),
                known_facts=data.get("known_facts", []),
                confidence=min(1.0, len(data.get("known_facts", [])) * _FACT_CONFIDENCE_WEIGHT + _BASE_DOMAIN_CONFIDENCE),
            )
            profile.life_domains.append(dp)

        profile.evidence_refs = evidence
        profile.confidence = self._estimate_confidence(profile)

        # Detect gaps
        self._detect_gaps(profile)

        # Build a short summary
        profile.summary = self._build_summary(profile)

    async def _find_nodes(self, user_id: str, node_type: str) -> list[Any]:
        """Attempt to retrieve nodes of *node_type* from the storage."""
        try:
            if hasattr(self._storage, "find_nodes"):
                return await self._storage.find_nodes(
                    user_id=user_id,
                    node_type=node_type,
                    limit=100,
                )
        except Exception as exc:
            logger.debug(
                "IdentityProfileBuilder: find_nodes(%s, %s) failed: %s",
                user_id,
                node_type,
                exc,
            )
        return []

    def _ingest_node(
        self,
        profile: IdentityProfile,
        node: Any,
        node_type: str,
        domain_data: dict[str, dict[str, Any]],
    ) -> None:
        """Extract useful fields from a graph node into the profile."""
        # Support both dataclass Node and plain dict
        if hasattr(node, "metadata"):
            meta = node.metadata or {}
            name = getattr(node, "name", None) or meta.get("name", "")
            text = getattr(node, "text", None) or meta.get("text", "")
        else:
            meta = node.get("metadata", {}) or {}
            name = node.get("name", "") or meta.get("name", "")
            text = node.get("text", "") or meta.get("text", "")

        domain = meta.get("domain", _FALLBACK_DOMAIN)
        bucket = domain_data.setdefault(domain, {"goals": [], "known_facts": [], "summary": ""})

        if node_type == "PROJECT":
            if name:
                bucket["goals"].append(name)
                profile.active_goals.append(name)
        elif node_type == "TASK":
            if name:
                bucket["known_facts"].append(f"task: {name}")
        elif node_type in ("BELIEF", "INSIGHT"):
            snippet = text or name
            if snippet:
                bucket["known_facts"].append(snippet[:120])
        elif node_type == "NEED":
            snippet = name or text
            if snippet:
                bucket["known_facts"].append(f"need: {snippet}")
        elif node_type == "VALUE":
            snippet = name or text
            if snippet and snippet not in profile.values:
                profile.values.append(snippet)

    def _detect_gaps(self, profile: IdentityProfile) -> None:
        """Add ProfileGap entries for obvious missing information."""
        user_id = profile.user_id

        if not profile.roles:
            profile.gaps.append(
                ProfileGap(
                    user_id=user_id,
                    domain="identity",
                    field_name="roles",
                    reason="No roles have been identified yet.",
                    priority=1,
                    suggested_question="What are the main roles you occupy in your life right now (e.g. engineer, parent, student)?",
                )
            )

        if not profile.active_goals:
            profile.gaps.append(
                ProfileGap(
                    user_id=user_id,
                    domain="goals",
                    field_name="active_goals",
                    reason="No active goals found in memory.",
                    priority=1,
                    suggested_question="What are your most important goals right now?",
                )
            )

        if not profile.values:
            profile.gaps.append(
                ProfileGap(
                    user_id=user_id,
                    domain="values",
                    field_name="values",
                    reason="No values found in memory.",
                    priority=2,
                    suggested_question="What values guide your decisions most strongly?",
                )
            )

        if not profile.life_domains:
            profile.gaps.append(
                ProfileGap(
                    user_id=user_id,
                    domain="general",
                    field_name="life_domains",
                    reason="No life-domain information has been captured yet.",
                    priority=2,
                    suggested_question="Which areas of your life are most important to you right now (career, health, relationships, …)?",
                )
            )

        # Low-confidence domains
        for dp in profile.life_domains:
            if dp.confidence < 0.3:
                profile.gaps.append(
                    ProfileGap(
                        user_id=user_id,
                        domain=dp.domain,
                        field_name="summary",
                        reason=f"Domain '{dp.domain}' has low confidence ({dp.confidence:.2f}).",
                        priority=2,
                        suggested_question=f"Can you tell me more about your situation in the '{dp.domain}' area of your life?",
                    )
                )

    def _add_bootstrap_gaps(self, profile: IdentityProfile) -> None:
        """Add minimal bootstrap gaps when no graph data is available."""
        self._detect_gaps(profile)

    @staticmethod
    def _estimate_confidence(profile: IdentityProfile) -> float:
        """Rough completeness score based on populated fields."""
        score = 0.0
        if profile.roles:
            score += 0.2
        if profile.active_goals:
            score += 0.2
        if profile.values:
            score += 0.2
        if profile.life_domains:
            score += 0.2
        if profile.skills:
            score += 0.1
        if profile.preferences:
            score += 0.1
        return round(min(1.0, score), 3)

    @staticmethod
    def _build_summary(profile: IdentityProfile) -> str:
        """Build a one-line summary from available data."""
        parts: list[str] = []
        if profile.active_goals:
            goals_str = ", ".join(profile.active_goals[:3])
            parts.append(f"goals: {goals_str}")
        if profile.values:
            values_str = ", ".join(profile.values[:3])
            parts.append(f"values: {values_str}")
        if not parts:
            return "No profile data available yet."
        return "Profile summary — " + "; ".join(parts) + "."
