"""IFS Part agents for the InnerCouncil debate system.

Each agent represents an Internal Family Systems part and produces a
:class:`DebateEntry` given the shared :class:`AgentContext` and the
debate log accumulated so far.
"""

from __future__ import annotations

import logging

from agents.ifs.models import DebateEntry
from core.pipeline.orchestrator import AgentContext

logger = logging.getLogger(__name__)


class IFSPartAgent:
    """Base agent for an IFS psyche part.

    Subclasses set *part_type* and *voice_prompt* to define the voice of the
    part.  The :meth:`deliberate` method returns a :class:`DebateEntry` that
    captures the part's position, emotional tone, and underlying need.
    """

    part_type: str = "SELF"
    voice_prompt: str = ""

    async def deliberate(
        self,
        context: AgentContext,
        council_log: list[DebateEntry],
    ) -> DebateEntry:
        """Produce a debate entry based on *context* and prior *council_log*.

        The default implementation uses keyword heuristics.  Subclasses may
        override to integrate LLM-based reasoning.
        """
        raise NotImplementedError  # pragma: no cover


class CriticAgent(IFSPartAgent):
    """Manager / inner-critic part — demands standards and control."""

    part_type = "MANAGER"
    voice_prompt = (
        "Ты — внутренний Критик (Менеджер). "
        "Твоя роль — защищать от провала и позора. "
        "Ты требуешь высоких стандартов и указываешь на ошибки, "
        "чтобы человек оставался в безопасности. "
        "Говори строго, но с заботой."
    )

    async def deliberate(
        self,
        context: AgentContext,
        council_log: list[DebateEntry],
    ) -> DebateEntry:
        text_lower = context.text.lower()
        signals = ["не смог", "не могу", "опять", "снова", "ненавижу себя", "провал", "ошибка"]
        intensity = sum(1 for s in signals if s in text_lower)
        confidence = min(0.3 + intensity * 0.15, 1.0)
        position = (
            "Нужно собраться и перестать допускать ошибки. "
            "Стандарты существуют не просто так."
        ) if intensity else "Пока нет оснований для критики."
        return DebateEntry(
            part_type=self.part_type,
            position=position,
            emotion="строгость" if intensity else "нейтральность",
            need="контроль" if intensity else None,
            confidence=confidence,
        )


class FirefighterAgent(IFSPartAgent):
    """Firefighter part — protects through avoidance and distraction."""

    part_type = "FIREFIGHTER"
    voice_prompt = (
        "Ты — Пожарный (Firefighter). "
        "Твоя роль — срочно снять боль и перенапряжение. "
        "Ты предлагаешь отвлечься, сбежать, получить мгновенное облегчение. "
        "Говори с пониманием и мягкостью."
    )

    async def deliberate(
        self,
        context: AgentContext,
        council_log: list[DebateEntry],
    ) -> DebateEntry:
        text_lower = context.text.lower()
        signals = [
            "залип", "отвлекаюсь", "убегаю", "игнорирую", "прокрастин",
            "пью", "ем", "скролл", "игр",
        ]
        intensity = sum(1 for s in signals if s in text_lower)
        confidence = min(0.3 + intensity * 0.15, 1.0)
        position = (
            "Сейчас важнее всего снять напряжение. "
            "Можно дать себе передышку — это не слабость."
        ) if intensity else "Не вижу острой необходимости в срочном облегчении."
        return DebateEntry(
            part_type=self.part_type,
            position=position,
            emotion="защита" if intensity else "спокойствие",
            need="безопасность" if intensity else None,
            confidence=confidence,
        )


class ExileAgent(IFSPartAgent):
    """Exile part — carries vulnerability, fear, and shame."""

    part_type = "EXILE"
    voice_prompt = (
        "Ты — Изгнанник (Exile). "
        "Ты несёшь ранимость, боль и стыд, "
        "которые другие части пытаются скрыть. "
        "Ты хочешь быть услышанным и принятым. "
        "Говори тихо и честно."
    )

    async def deliberate(
        self,
        context: AgentContext,
        council_log: list[DebateEntry],
    ) -> DebateEntry:
        text_lower = context.text.lower()
        signals = [
            "боюсь", "стыжусь", "одинок", "никто не", "не достоин",
            "больно", "страшно", "стыд", "вина",
        ]
        intensity = sum(1 for s in signals if s in text_lower)
        confidence = min(0.3 + intensity * 0.15, 1.0)
        position = (
            "Мне больно и страшно. Хочу чтобы кто-то просто был рядом. "
            "Принятие важнее достижений."
        ) if intensity else "Сейчас я в безопасности."
        return DebateEntry(
            part_type=self.part_type,
            position=position,
            emotion="уязвимость" if intensity else "покой",
            need="принятие" if intensity else None,
            confidence=confidence,
        )


class SelfAgent(IFSPartAgent):
    """Self (witness/integrator) — observes all parts and synthesises."""

    part_type = "SELF"
    voice_prompt = (
        "Ты — Самость (Self). "
        "Ты наблюдаешь все части с состраданием и без осуждения. "
        "Ты видишь потребности каждой части и ищешь баланс. "
        "Говори спокойно, мудро и с сочувствием."
    )

    async def deliberate(
        self,
        context: AgentContext,
        council_log: list[DebateEntry],
    ) -> DebateEntry:
        active = [e for e in council_log if e.confidence > 0.4]
        needs: list[str] = [e.need for e in active if e.need]
        parts_active = [e.part_type for e in active]

        if not active:
            return DebateEntry(
                part_type=self.part_type,
                position="Все части спокойны. Можно двигаться дальше осознанно.",
                emotion="ясность",
                need=None,
                confidence=0.8,
            )

        needs_str = ", ".join(dict.fromkeys(needs)) if needs else "внимание"
        parts_str = ", ".join(dict.fromkeys(parts_active))
        position = (
            f"Вижу активные части: {parts_str}. "
            f"Потребности сейчас: {needs_str}. "
            "Важно дать каждой части быть услышанной, прежде чем действовать."
        )
        return DebateEntry(
            part_type=self.part_type,
            position=position,
            emotion="сострадание",
            need=needs[0] if needs else None,
            confidence=0.9,
        )


def build_default_parts() -> dict[str, IFSPartAgent]:
    """Return the standard set of IFS part agents."""
    agents: list[IFSPartAgent] = [
        CriticAgent(),
        FirefighterAgent(),
        ExileAgent(),
        SelfAgent(),
    ]
    return {a.part_type: a for a in agents}
