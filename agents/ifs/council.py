"""InnerCouncil — IFS-based multi-agent deliberation for SELF-OS.

Runs a two-round deliberation among four IFS part agents before the system
produces a final reply.  This mirrors the Stage-3 "Society of Mind" design
from ``docs/FRONTIER_VISION_REPORT.md``::

    Round 1 — each part voices its perspective independently.
    Round 2 — Self synthesises a compassionate, integrated position.

Usage::

    council = InnerCouncil()
    result = await council.deliberate(context)
    # result.synthesis — the Self-led integrated response fragment
    # result.voices    — list of (role, voice) pairs from round 1

The council integrates with the pipeline by being called from
:class:`~core.pipeline.orchestrator.AgentOrchestrator` for sessions where
``session_conflict=True`` or when active parts are present.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from agents.ifs.parts import (
    CriticAgent,
    ExileAgent,
    FirefighterAgent,
    IFSAgentContext,
    IFSAgentResult,
    SelfAgent,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# DTO
# ---------------------------------------------------------------------------


@dataclass
class CouncilResult:
    """Output of a full InnerCouncil deliberation."""

    voices: list[IFSAgentResult] = field(default_factory=list)
    """Round-1 voiced perspectives from Critic, Firefighter, Exile."""

    synthesis: str = ""
    """Round-2 Self-led integrated summary."""

    dominant_need: str = ""
    """The most commonly signalled underlying need across all parts."""

    recommended_modality: str = ""
    """Suggested therapeutic modality (e.g. 'IFS_parts_dialogue', 'CBT_reframe')."""


# ---------------------------------------------------------------------------
# InnerCouncil
# ---------------------------------------------------------------------------


class InnerCouncil:
    """Orchestrates a 2-round IFS deliberation.

    Round 1: Critic, Firefighter, and Exile each voice their perspective.
    Round 2: Self synthesises an integrated, compassionate position.

    Parameters
    ----------
    agents:
        Override the default agent set (useful for testing).
    """

    def __init__(
        self,
        agents: dict | None = None,
    ) -> None:
        if agents is not None:
            self._agents = agents
        else:
            critic = CriticAgent()
            firefighter = FirefighterAgent()
            exile = ExileAgent()
            self._agents = {
                "critic": critic,
                "firefighter": firefighter,
                "exile": exile,
            }
        self._self_agent = SelfAgent()

    async def deliberate(self, context: IFSAgentContext) -> CouncilResult:
        """Run the two-round deliberation and return a :class:`CouncilResult`.

        Parameters
        ----------
        context:
            Shared context including the user message, intent, mood, and
            existing parts context.
        """
        # ---- Round 1: gather part voices -----------------------------------
        voices: list[IFSAgentResult] = []
        for role, agent in self._agents.items():
            try:
                result = await agent.respond(context)
                voices.append(result)
                logger.debug("InnerCouncil round-1 — %s: %r", role, result.voice[:60])
            except Exception as exc:
                logger.warning("InnerCouncil: agent %r failed — %s", role, exc)

        # ---- Round 2: re-run with council awareness -------------------------
        # Each agent gets to see what others said in Round 1 and may adjust.
        round2_voices: list[IFSAgentResult] = []
        for role, agent in self._agents.items():
            try:
                result = await agent.respond(context, council_voices=voices)
                round2_voices.append(result)
                logger.debug("InnerCouncil round-2 — %s: %r", role, result.voice[:60])
            except TypeError:
                # Agent doesn't support council_voices yet — fall back to Round 1 result
                round1 = next((v for v in voices if v.part_role == role), None)
                if round1:
                    round2_voices.append(round1)
            except Exception as exc:
                logger.warning("InnerCouncil round-2: agent %r failed — %s", role, exc)

        final_voices = round2_voices if round2_voices else voices

        # ---- Determine dominant need ---------------------------------------
        need_counts: dict[str, int] = {}
        for v in final_voices:
            if v.need:
                need_counts[v.need] = need_counts.get(v.need, 0) + 1
        dominant_need = max(need_counts, key=lambda k: need_counts[k]) if need_counts else ""

        # ---- Inject round-2 voices into Self's context --------------------
        enriched_parts = list(context.parts_context)
        for v in final_voices:
            enriched_parts.append({"subtype": v.part_role, "voice": v.voice})

        self_ctx = IFSAgentContext(
            user_id=context.user_id,
            text=context.text,
            intent=context.intent,
            mood_context=context.mood_context,
            parts_context=enriched_parts,
            graph_context=context.graph_context,
        )

        # ---- Self synthesis (Round 3 effectively) --------------------------
        self_result = await self._self_agent.respond(self_ctx)
        logger.debug("InnerCouncil self-synthesis: %r", self_result.voice[:80])

        # ---- Pick recommended modality ------------------------------------
        modality = _pick_modality(final_voices, context)

        return CouncilResult(
            voices=final_voices,
            synthesis=self_result.voice,
            dominant_need=dominant_need,
            recommended_modality=modality,
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _pick_modality(voices: list[IFSAgentResult], context: IFSAgentContext) -> str:
    """Heuristically pick a therapy modality from the council voices."""
    text_lower = context.text.lower()

    # Somatic grounding takes priority for physical/body signals
    somatic_keywords = ["тело", "грудь", "сжатие", "дыхание", "сердце", "живот"]
    if any(kw in text_lower for kw in somatic_keywords):
        return "somatic_grounding"

    # Check which parts were actually triggered (non-idle voice)
    active_roles = {v.part_role for v in voices if v.voice}

    # IFS dialogue when two or more distinct parts active
    if len(active_roles) >= 2:
        return "IFS_parts_dialogue"

    # CBT reframe when critic is active (distortions)
    if "critic" in active_roles:
        return "CBT_reframe"

    # Empathic validation when exile is active (shame / loneliness)
    if "exile" in active_roles:
        return "empathic_validation"

    # ACT defusion for avoidance / firefighter
    if "firefighter" in active_roles:
        return "ACT_defusion"

    return "empathic_validation"
