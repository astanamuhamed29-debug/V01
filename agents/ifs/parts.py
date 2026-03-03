"""IFS (Internal Family Systems) Part agents for the InnerCouncil debate system.

Each *Part* is a sub-personality that analyses the user's message and produces a
position with a confidence score.  The :class:`InnerCouncil` runs two rounds:

- **Round 1**: all Parts analyse the raw message independently.
- **Round 2**: each Part re-evaluates, taking into account what other Parts said
  in Round 1 (council_log).  This is the actual debate — each Part adjusts its
  stance when it observes strong positions from peers.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from agents.ifs.signals import EMOTION_SIGNALS, PART_SIGNALS

if TYPE_CHECKING:
    from core.llm_client import LLMClient

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# DTOs
# ---------------------------------------------------------------------------


@dataclass
class PartPosition:
    """Position expressed by a single IFS Part during deliberation."""

    part_name: str
    position: str
    confidence: float        # 0.0 – 1.0
    needs: list[str] = field(default_factory=list)
    fears: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class CouncilVerdict:
    """Aggregated result produced by the full InnerCouncil debate."""

    dominant_part: str
    consensus_text: str
    positions: list[PartPosition] = field(default_factory=list)
    round1_log: list[PartPosition] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Agent context
# ---------------------------------------------------------------------------


@dataclass
class IFSContext:
    """Context passed to each Part agent."""

    user_id: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------


class IFSPartAgent:
    """Abstract base for an IFS Part agent.

    Subclasses implement :meth:`deliberate` to analyse the user message and
    return a :class:`PartPosition`.

    Parameters
    ----------
    llm_client:
        Optional LLM client.  When provided, :meth:`deliberate` may use
        :attr:`voice_prompt` as a system prompt to generate a richer
        response via the LLM.  When ``None``, the keyword-based fallback
        logic runs instead.
    """

    name: str = "base"
    voice_prompt: str = ""

    def __init__(self, llm_client: "LLMClient | None" = None) -> None:
        self._llm = llm_client

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _keyword_score(self, text: str, keywords: list[str]) -> float:
        """Return a simple keyword-hit score in [0, 1]."""
        t = text.lower()
        hits = sum(1 for kw in keywords if kw in t)
        return min(hits / max(len(keywords), 1), 1.0)

    async def _llm_deliberate(
        self,
        context: IFSContext,
        council_log: list[PartPosition],
    ) -> str | None:
        """Use LLM to generate a nuanced position using *voice_prompt*.

        Returns the generated text or ``None`` if the call fails / is
        unavailable.
        """
        if self._llm is None:
            return None
        try:
            council_summary = ""
            if council_log:
                lines = [
                    f"{p.part_name} (confidence={p.confidence:.2f}): {p.position}"
                    for p in council_log
                ]
                council_summary = "\n".join(lines)

            prompt_parts = [f"Сообщение пользователя: {context.text}"]
            if council_summary:
                prompt_parts.append(
                    f"\nМнения других частей:\n{council_summary}"
                )
            prompt_parts.append(
                "\nВыскажи свою позицию кратко (1-2 предложения)."
            )

            response = await self._llm.complete(
                system=self.voice_prompt,
                user="\n".join(prompt_parts),
            )
            return response.strip() if response else None
        except Exception as exc:
            logger.debug("LLM deliberate failed for %s: %s", self.name, exc)
            return None

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    async def deliberate(
        self,
        context: IFSContext,
        council_log: list[PartPosition] | None = None,
    ) -> PartPosition:
        """Produce a :class:`PartPosition` for *context*.

        Subclasses should override this method.  The base implementation
        returns a neutral position.

        Parameters
        ----------
        context:
            The user's message context.
        council_log:
            Positions from other Parts expressed in Round 1.  Subclasses
            use this to adjust their stance in Round 2.
        """
        return PartPosition(
            part_name=self.name,
            position="Наблюдаю.",
            confidence=0.3,
        )


# ---------------------------------------------------------------------------
# Concrete Part agents
# ---------------------------------------------------------------------------


class CriticAgent(IFSPartAgent):
    """The inner Critic — holds standards and points out failures.

    In Round 2, if ExileAgent expressed high pain (confidence > 0.5), the
    Critic softens its stance to avoid overwhelming the Exile.
    """

    name = "Критик"
    voice_prompt = (
        "Ты — внутренний Критик. Твоя роль: указывать на ошибки и зоны роста, "
        "поддерживать высокие стандарты. Говори прямо, но не жестоко."
    )

    async def deliberate(
        self,
        context: IFSContext,
        council_log: list[PartPosition] | None = None,
    ) -> PartPosition:
        """Evaluate the message; soften if Exile expressed high pain."""
        text = context.text

        # Try LLM first
        llm_text = await self._llm_deliberate(context, council_log or [])

        # Keyword-based confidence
        critic_keywords = PART_SIGNALS.get("CRITIC", [])
        base_conf = max(0.2, self._keyword_score(text, critic_keywords) + 0.3)

        position = llm_text or (
            "Я замечаю, что здесь есть пространство для улучшения. "
            "Важно быть честным с собой."
        )

        # Round 2 adjustment: if Exile has high pain, soften
        if council_log:
            exile_positions = [p for p in council_log if p.part_name == "Изгнанник"]
            if exile_positions and exile_positions[0].confidence > 0.5:
                base_conf = max(0.1, base_conf - 0.25)
                if not llm_text:
                    position = (
                        "Вижу, что сейчас не время для критики. "
                        "Сначала нужно позаботиться о боли."
                    )

        return PartPosition(
            part_name=self.name,
            position=position,
            confidence=min(base_conf, 1.0),
            needs=["качество", "совершенство"],
            fears=["провал", "осуждение"],
        )


class FirefighterAgent(IFSPartAgent):
    """The Firefighter — acts impulsively to douse emotional pain.

    In Round 2, if ExileAgent has high activation, urgency increases.
    """

    name = "Пожарный"
    voice_prompt = (
        "Ты — Пожарный. Твоя роль: срочно погасить боль любым способом. "
        "Действуй быстро, даже если это создаёт новые проблемы."
    )

    async def deliberate(
        self,
        context: IFSContext,
        council_log: list[PartPosition] | None = None,
    ) -> PartPosition:
        """Evaluate urgency; increase if Exile is highly activated."""
        text = context.text

        llm_text = await self._llm_deliberate(context, council_log or [])

        ff_keywords = PART_SIGNALS.get("FIREFIGHTER", [])
        base_conf = max(0.15, self._keyword_score(text, ff_keywords) + 0.2)

        position = llm_text or (
            "Нужно срочно что-то сделать, чтобы стало лучше прямо сейчас."
        )

        # Round 2: if Exile has high activation, Firefighter urgency increases
        if council_log:
            exile_positions = [p for p in council_log if p.part_name == "Изгнанник"]
            if exile_positions and exile_positions[0].confidence > 0.5:
                base_conf = min(1.0, base_conf + 0.3)
                if not llm_text:
                    position = (
                        "Боль нарастает. Мне нужно действовать немедленно, "
                        "чтобы облегчить это страдание."
                    )

        return PartPosition(
            part_name=self.name,
            position=position,
            confidence=min(base_conf, 1.0),
            needs=["безопасность", "облегчение"],
            fears=["боль", "страдание"],
        )


class ExileAgent(IFSPartAgent):
    """The Exile — carries the original wound and unmet needs.

    In Round 2, if Critic dominated, the Exile's need for safety increases.
    """

    name = "Изгнанник"
    voice_prompt = (
        "Ты — Изгнанник. Ты несёшь старую боль и невыполненные потребности. "
        "Говори об уязвимости, страхе быть отвергнутым, глубокой нужде в любви."
    )

    async def deliberate(
        self,
        context: IFSContext,
        council_log: list[PartPosition] | None = None,
    ) -> PartPosition:
        """Evaluate vulnerability; boost safety need if Critic dominated."""
        text = context.text

        llm_text = await self._llm_deliberate(context, council_log or [])

        exile_keywords = PART_SIGNALS.get("EXILE", [])
        emotion_pain = EMOTION_SIGNALS.get("стыд", []) + EMOTION_SIGNALS.get("вина", [])
        base_conf = max(0.1, self._keyword_score(text, exile_keywords + emotion_pain) + 0.15)

        position = llm_text or (
            "Мне больно. Я чувствую себя одиноким и непонятым."
        )

        # Round 2: if Critic dominated, Exile's need for safety increases
        if council_log:
            critic_positions = [p for p in council_log if p.part_name == "Критик"]
            if critic_positions and critic_positions[0].confidence > 0.5:
                base_conf = min(1.0, base_conf + 0.3)
                if not llm_text:
                    position = (
                        "Критик давит на меня. Мне нужна безопасность и принятие, "
                        "а не ещё больше осуждения."
                    )

        return PartPosition(
            part_name=self.name,
            position=position,
            confidence=min(base_conf, 1.0),
            needs=["принятие", "безопасность", "любовь"],
            fears=["отвержение", "осуждение", "одиночество"],
        )


class SelfAgent(IFSPartAgent):
    """The Self — the compassionate, curious center that can lead healing.

    Uses council_log to synthesise a balanced response from all Parts.
    """

    name = "Самость"
    voice_prompt = (
        "Ты — Самость. Ты мудрый, спокойный, сострадательный наблюдатель. "
        "Видишь все части, понимаешь их потребности, ведёшь к исцелению."
    )

    async def deliberate(
        self,
        context: IFSContext,
        council_log: list[PartPosition] | None = None,
    ) -> PartPosition:
        """Synthesise a balanced Self response informed by council_log."""
        text = context.text

        llm_text = await self._llm_deliberate(context, council_log or [])

        self_keywords = PART_SIGNALS.get("SELF", [])
        base_conf = max(0.4, self._keyword_score(text, self_keywords) + 0.4)

        if llm_text:
            position = llm_text
        elif council_log:
            # Synthesise from what the other parts expressed
            pain_parts = [
                p.part_name
                for p in council_log
                if p.part_name == "Изгнанник" and p.confidence > 0.4
            ]
            urgent_parts = [
                p.part_name
                for p in council_log
                if p.part_name == "Пожарный" and p.confidence > 0.5
            ]

            if pain_parts:
                position = (
                    "Я слышу боль внутри. Давай с состраданием посмотрим "
                    "на то, что происходит, и найдём путь к исцелению."
                )
                base_conf = min(1.0, base_conf + 0.1)
            elif urgent_parts:
                position = (
                    "Замечаю желание действовать прямо сейчас. "
                    "Давай сделаем паузу и поймём, что действительно нужно."
                )
            else:
                position = (
                    "Я с тобой. Давай вместе разберёмся с тем, что происходит."
                )
        else:
            position = (
                "Я слышу тебя. С любопытством и состраданием смотрю на ситуацию."
            )

        return PartPosition(
            part_name=self.name,
            position=position,
            confidence=min(base_conf, 1.0),
            needs=["ясность", "исцеление", "связь"],
            fears=[],
        )


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def build_default_parts(
    llm_client: "LLMClient | None" = None,
) -> list[IFSPartAgent]:
    """Return the default set of four IFS Part agents.

    Parameters
    ----------
    llm_client:
        Optional LLM client passed to each agent.  When provided, agents
        will use the LLM to generate nuanced responses via their
        ``voice_prompt``.
    """
    return [
        CriticAgent(llm_client=llm_client),
        FirefighterAgent(llm_client=llm_client),
        ExileAgent(llm_client=llm_client),
        SelfAgent(llm_client=llm_client),
    ]
