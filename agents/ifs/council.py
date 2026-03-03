"""InnerCouncil — two-round IFS debate orchestrator.

Round 1: all Part agents analyse the user's message independently.
Round 2: each Part re-evaluates, incorporating what other Parts said,
         producing genuinely different output (not a repeat of Round 1).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from agents.ifs.parts import (
    CouncilVerdict,
    IFSContext,
    IFSPartAgent,
    PartPosition,
    build_default_parts,
)

if TYPE_CHECKING:
    from core.llm_client import LLMClient

logger = logging.getLogger(__name__)


class InnerCouncil:
    """Orchestrates a two-round IFS debate between Part agents.

    Parameters
    ----------
    parts:
        List of :class:`IFSPartAgent` instances to include in the debate.
        Defaults to the four standard parts built by
        :func:`build_default_parts`.
    llm_client:
        Optional LLM client forwarded to :func:`build_default_parts` when
        *parts* is not explicitly provided.
    """

    def __init__(
        self,
        parts: list[IFSPartAgent] | None = None,
        llm_client: "LLMClient | None" = None,
    ) -> None:
        self._parts: list[IFSPartAgent] = (
            parts if parts is not None else build_default_parts(llm_client=llm_client)
        )

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    async def deliberate(self, context: IFSContext) -> CouncilVerdict:
        """Run the two-round IFS debate for *context*.

        Returns
        -------
        CouncilVerdict
            Aggregated result with dominant Part, consensus text, and
            the full Round-2 positions.
        """
        # ── Round 1 — independent analysis ───────────────────────────
        round1: list[PartPosition] = []
        for part in self._parts:
            try:
                pos = await part.deliberate(context, council_log=None)
                round1.append(pos)
            except Exception as exc:
                logger.warning("Part %r failed in Round 1: %s", part.name, exc)

        # ── Round 2 — debate aware of peers' Round-1 positions ────────
        round2: list[PartPosition] = []
        for part in self._parts:
            try:
                pos = await part.deliberate(context, council_log=round1)
                round2.append(pos)
            except Exception as exc:
                logger.warning("Part %r failed in Round 2: %s", part.name, exc)

        # ── Determine dominant part (highest Round-2 confidence) ──────
        dominant = max(round2, key=lambda p: p.confidence) if round2 else None
        dominant_name = dominant.part_name if dominant else "Самость"

        # ── Build consensus text ──────────────────────────────────────
        consensus = self._build_consensus(round2, dominant_name)

        return CouncilVerdict(
            dominant_part=dominant_name,
            consensus_text=consensus,
            positions=round2,
            round1_log=round1,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_consensus(
        self,
        positions: list[PartPosition],
        dominant_name: str,
    ) -> str:
        """Combine all Part positions into a short consensus statement."""
        if not positions:
            return "Совет частей не смог прийти к выводу."

        lines: list[str] = []
        for pos in positions:
            marker = "★" if pos.part_name == dominant_name else "·"
            lines.append(f"{marker} {pos.part_name}: {pos.position}")

        return "\n".join(lines)
