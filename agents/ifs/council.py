"""InnerCouncil — multi-agent IFS debate orchestrator.

Implements a 2-round deliberation pattern inspired by
*Du et al., "Improving Factuality and Reasoning in LLMs through
Multi-Agent Debate" (2023)*.

Activation condition:
    ``session_conflict=True`` in graph context **or** 2+ active parts in
    ``parts_context``.

Round 1 (parallel): each IFS part receives the user message and graph
    context, then forms an initial position.
Round 2 (debate): parts see each other's positions and may adjust their
    stance.
Final synthesis: the :class:`SelfAgent` integrates all positions into a
    single :class:`CouncilVerdict`.
"""

from __future__ import annotations

import logging
from typing import Any

from agents.ifs.models import CouncilVerdict, DebateEntry
from agents.ifs.parts import IFSPartAgent, SelfAgent, build_default_parts
from core.pipeline.orchestrator import AgentContext

logger = logging.getLogger(__name__)


class InnerCouncil:
    """Orchestrate an internal IFS-parts dialogue before response generation.

    Usage::

        council = InnerCouncil()
        verdict = await council.deliberate(context, active_parts)
    """

    def __init__(
        self,
        parts: dict[str, IFSPartAgent] | None = None,
    ) -> None:
        self._parts: dict[str, IFSPartAgent] = parts if parts is not None else build_default_parts()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def deliberate(
        self,
        context: AgentContext,
        active_parts: list[IFSPartAgent] | None = None,
        rounds: int = 2,
    ) -> CouncilVerdict:
        """Run a multi-round debate and return the synthesised verdict.

        Parameters
        ----------
        context:
            Shared agent context for the current pipeline turn.
        active_parts:
            Subset of IFS parts to include.  When *None* all registered
            parts participate.
        rounds:
            Number of deliberation rounds (default 2).
        """
        parts = active_parts or list(self._parts.values())
        if not parts:
            return self._empty_verdict()

        # Ensure SelfAgent participates last as integrator
        non_self = [p for p in parts if not isinstance(p, SelfAgent)]
        self_agents = [p for p in parts if isinstance(p, SelfAgent)]
        if not self_agents:
            self_agents = [SelfAgent()]

        log: list[DebateEntry] = []

        # --- Round 1: independent positions ---
        for part in non_self:
            try:
                entry = await part.deliberate(context, council_log=[])
                log.append(entry)
            except Exception as exc:
                logger.warning("Part %s failed in round 1: %s", part.part_type, exc)

        # --- Additional rounds: parts see prior log ---
        for _round in range(1, rounds):
            updated: list[DebateEntry] = []
            for part in non_self:
                try:
                    entry = await part.deliberate(context, council_log=log)
                    updated.append(entry)
                except Exception as exc:
                    logger.warning(
                        "Part %s failed in round %d: %s", part.part_type, _round + 1, exc,
                    )
            log.extend(updated)

        # --- Self synthesis ---
        self_agent = self_agents[0]
        try:
            self_entry = await self_agent.deliberate(context, council_log=log)
            log.append(self_entry)
        except Exception as exc:
            logger.warning("SelfAgent synthesis failed: %s", exc)
            self_entry = DebateEntry(
                part_type="SELF",
                position="Не удалось провести синтез.",
                emotion="неопределённость",
                confidence=0.3,
            )
            log.append(self_entry)

        return self._build_verdict(log, self_entry)

    # ------------------------------------------------------------------
    # Activation check (static helper)
    # ------------------------------------------------------------------

    @staticmethod
    def should_activate(
        graph_context: dict[str, Any],
        parts_context: list[dict[str, Any]],
    ) -> bool:
        """Return *True* when the InnerCouncil should be invoked.

        Conditions (either one triggers activation):
        * ``session_conflict`` flag set in *graph_context*
        * 2 or more active parts present in *parts_context*
        """
        if graph_context.get("session_conflict"):
            return True
        return len(parts_context) >= 2

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _build_verdict(
        self,
        log: list[DebateEntry],
        self_entry: DebateEntry,
    ) -> CouncilVerdict:
        """Derive the final verdict from the debate log."""
        # Dominant part = highest confidence among non-SELF entries
        non_self = [e for e in log if e.part_type != "SELF"]
        dominant = max(non_self, key=lambda e: e.confidence) if non_self else self_entry

        # Unresolved conflict when multiple parts have high confidence
        high_conf = [e for e in non_self if e.confidence >= 0.5]
        unique_parts = {e.part_type for e in high_conf}
        unresolved = len(unique_parts) >= 2

        return CouncilVerdict(
            dominant_part=dominant.part_type,
            consensus_reply=self_entry.position,
            unresolved_conflict=unresolved,
            internal_log=log,
        )

    @staticmethod
    def _empty_verdict() -> CouncilVerdict:
        return CouncilVerdict(
            dominant_part="SELF",
            consensus_reply="",
            unresolved_conflict=False,
            internal_log=[],
        )
