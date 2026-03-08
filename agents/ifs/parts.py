"""IFS (Internal Family Systems) part agents for the InnerCouncil.

Each agent represents one IFS role and produces a short voiced perspective
on the current situation.  All agents share the same lightweight interface::

    result = await agent.respond(context)

Agents are **pure** (no I/O side-effects) and intentionally simple — they
apply keyword heuristics so they work without an LLM call.  When an LLM
client is injected the agent can optionally escalate to a richer response.

Roles (IFS model):
    CriticAgent       — inner Critic / Protector Manager.
    FirefighterAgent  — impulsive Firefighter / escape protector.
    ExileAgent        — wounded Exile / inner child.
    SelfAgent         — the compassionate Self / observer.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# DTOs
# ---------------------------------------------------------------------------


@dataclass
class IFSAgentContext:
    """Input context shared with every IFS part agent."""

    user_id: str
    text: str
    intent: str
    mood_context: dict[str, Any] = field(default_factory=dict)
    parts_context: list[dict[str, Any]] = field(default_factory=list)
    graph_context: dict[str, Any] = field(default_factory=dict)


@dataclass
class IFSAgentResult:
    """Perspective voiced by a single IFS part agent."""

    part_role: str        # "critic" | "firefighter" | "exile" | "self"
    voice: str            # Short voiced statement (1-3 sentences)
    need: str = ""        # Underlying need this part is protecting
    recommendation: str = ""  # What this part suggests


# ---------------------------------------------------------------------------
# Base
# ---------------------------------------------------------------------------


class _BaseIFSAgent:
    role: str = "base"

    async def respond(self, context: IFSAgentContext) -> IFSAgentResult:  # pragma: no cover
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Specialised agents
# ---------------------------------------------------------------------------


class CriticAgent(_BaseIFSAgent):
    """Inner Critic / Manager.

    Activates on self-blame, distortions, perfectionism signals.
    Goal: protect the person from failure via high standards.
    """

    role = "critic"

    _TRIGGERS = [
        "подвёл", "подвела", "ненавижу себя", "снова", "опять",
        "неудача", "провал", "недостаточно", "must", "should",
        "надо было", "не смог", "не смогла", "облажался", "облажалась",
    ]

    async def respond(
        self,
        context: IFSAgentContext,
        council_voices: list[IFSAgentResult] | None = None,
    ) -> IFSAgentResult:
        """Voice the Critic's perspective.

        In Round 2, if ``council_voices`` contains an active ExileAgent
        position, the Critic softens to avoid piling on pain.
        """
        text_lower = context.text.lower()
        activated = any(t in text_lower for t in self._TRIGGERS)

        if activated:
            # Round 2 adjustment: soften if Exile is in high-pain mode
            if council_voices:
                exile_active = any(
                    v.part_role == "exile"
                    and any(
                        kw in v.voice.lower()
                        for kw in ("боль", "страшно", "больно", "одинок")
                    )
                    for v in council_voices
                )
                if exile_active:
                    return IFSAgentResult(
                        part_role=self.role,
                        voice=(
                            "Вижу, что сейчас не время для критики. "
                            "Чувствую боль и хочу её поддержать, а не осуждать."
                        ),
                        need="защита через принятие",
                        recommendation="Отступить и дать пространство для исцеления.",
                    )

            voice = (
                "Ты снова повторяешь ту же ошибку. "
                "Мне нужно убедиться, что ты соответствуешь стандартам — "
                "иначе всё рухнет."
            )
            need = "стабильность и предсказуемость"
            rec = "Установить чёткие стандарты и систему контроля."
        else:
            voice = "Пока всё под контролем, продолжай двигаться по плану."
            need = "контроль"
            rec = "Продолжать текущий курс."

        return IFSAgentResult(
            part_role=self.role, voice=voice, need=need, recommendation=rec
        )


class FirefighterAgent(_BaseIFSAgent):
    """Impulsive Firefighter / escape protector.

    Activates on avoidance, overwhelm, escapism signals.
    Goal: immediately relieve unbearable pain via distraction.
    """

    role = "firefighter"

    _TRIGGERS = [
        "залип", "избегаю", "отвлекаюсь", "прокрастинирую",
        "ушёл", "ушла", "не могу начать", "тяжело", "невыносимо",
        "сбежать", "игры", "соцсети", "netflix", "youtube",
    ]

    async def respond(
        self,
        context: IFSAgentContext,
        council_voices: list[IFSAgentResult] | None = None,
    ) -> IFSAgentResult:
        """Voice the Firefighter's perspective.

        In Round 2, if ``council_voices`` contains an active ExileAgent,
        the Firefighter's urgency increases.
        """
        text_lower = context.text.lower()
        activated = any(t in text_lower for t in self._TRIGGERS)

        if activated:
            # Round 2 adjustment: increase urgency when Exile is active
            if council_voices:
                exile_active = any(
                    v.part_role == "exile"
                    and any(
                        kw in v.voice.lower()
                        for kw in ("боль", "страшно", "больно", "одинок")
                    )
                    for v in council_voices
                )
                if exile_active:
                    return IFSAgentResult(
                        part_role=self.role,
                        voice=(
                            "Изгнанник страдает — мне нужно действовать немедленно. "
                            "Дать ему хоть какое-то облегчение прямо сейчас!"
                        ),
                        need="срочное облегчение боли",
                        recommendation="Краткосрочная разгрузка для снижения боли Изгнанника.",
                    )

            voice = (
                "Мне нужно было дать тебе передышку прямо сейчас — "
                "боль была слишком сильной, а ты не мог остановиться сам."
            )
            need = "мгновенное облегчение и отдых"
            rec = "Дать себе короткий осознанный перерыв вместо бессознательного побега."
        else:
            voice = "Напряжение терпимое, в побеге пока нет нужды."
            need = "безопасность"
            rec = "Оставаться в контакте с задачей."

        return IFSAgentResult(
            part_role=self.role, voice=voice, need=need, recommendation=rec
        )


class ExileAgent(_BaseIFSAgent):
    """Wounded Exile / inner child.

    Activates on shame, loneliness, fear of rejection signals.
    Goal: be seen, accepted, and loved.
    """

    role = "exile"

    _TRIGGERS = [
        "стыдно", "стыжусь", "одинок", "одинока", "никому не нужен",
        "никому не нужна", "боюсь", "отвергнут", "отвергнута",
        "не принимают", "плохой", "плохая", "недостоин", "недостойна",
    ]

    async def respond(
        self,
        context: IFSAgentContext,
        council_voices: list[IFSAgentResult] | None = None,
    ) -> IFSAgentResult:
        """Voice the Exile's perspective.

        In Round 2, if ``council_voices`` contains a dominant Critic,
        the Exile's need for safety increases.
        """
        text_lower = context.text.lower()
        activated = any(t in text_lower for t in self._TRIGGERS)

        if activated:
            # Round 2 adjustment: amplify safety need when Critic dominated
            if council_voices:
                critic_active = any(
                    v.part_role == "critic"
                    and any(
                        kw in v.voice.lower()
                        for kw in ("ошибку", "стандарт", "рухнет", "снова")
                    )
                    for v in council_voices
                )
                if critic_active:
                    return IFSAgentResult(
                        part_role=self.role,
                        voice=(
                            "Критик давит на меня. "
                            "Мне нужна безопасность и принятие, а не ещё больше осуждения."
                        ),
                        need="безопасность и защита от критики",
                        recommendation="Сначала создать безопасное пространство, потом работать над ошибками.",
                    )

            voice = (
                "Мне больно и страшно. Я просто хочу, чтобы меня увидели "
                "и приняли таким, какой я есть."
            )
            need = "принятие, любовь, безопасная привязанность"
            rec = "Признать эту боль и дать ей место, не убегая и не подавляя."
        else:
            voice = "Сейчас я в безопасности и чувствую себя принятым."
            need = "принятие"
            rec = "Поддерживать ощущение безопасности."

        return IFSAgentResult(
            part_role=self.role, voice=voice, need=need, recommendation=rec
        )


class SelfAgent(_BaseIFSAgent):
    """Compassionate Self / observer.

    Always responds; synthesises the voices of other parts with curiosity,
    compassion, clarity, and calmness (the 8 Cs of IFS Self-leadership).
    """

    role = "self"

    async def respond(self, context: IFSAgentContext) -> IFSAgentResult:
        mood = context.mood_context
        label = mood.get("dominant_label") or mood.get("label") or ""
        parts_info = context.parts_context

        if label:
            mood_note = f"Замечаю, что сейчас преобладает состояние «{label}»."
        else:
            mood_note = "Присутствую здесь полностью."

        if parts_info:
            part_names = [
                p.get("subtype") or p.get("key") or "часть"
                for p in parts_info[:2]
            ]
            parts_note = " Вижу активные части: " + " и ".join(part_names) + "."
        else:
            parts_note = ""

        voice = (
            f"{mood_note}{parts_note} "
            "Я здесь, чтобы слышать каждую часть с состраданием "
            "и помочь найти путь вперёд без осуждения."
        )

        return IFSAgentResult(
            part_role=self.role,
            voice=voice.strip(),
            need="интеграция и исцеление",
            recommendation=(
                "Дать каждой части голос, признать её намерение и выбрать "
                "осознанный ответ вместо автоматической реакции."
            ),
        )
