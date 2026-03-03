"""Proactive suggestions tool for SELF-OS.

Generates contextually relevant proactive suggestions based on the user's
current :class:`~core.psyche.state.PsycheState` and active
:class:`~core.goals.engine.Goal` list.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

from core.tools.base import Tool, ToolCallResult, ToolParameter

if TYPE_CHECKING:
    from core.goals.engine import Goal
    from core.llm.llm_client import LLMClient
    from core.psyche.state import PsycheState

logger = logging.getLogger(__name__)

SuggestionType = Literal["educational", "health", "productivity", "social", "creative"]

_SUGGESTION_TYPES: tuple[str, ...] = (
    "educational",
    "health",
    "productivity",
    "social",
    "creative",
)


@dataclass
class Suggestion:
    """A single proactive suggestion for the user."""

    type: str                   # educational | health | productivity | social | creative
    title: str
    body: str
    rationale: str = ""         # why this suggestion is relevant now
    priority: int = 3           # 1 = highest
    tags: list[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a plain dict (JSON-safe)."""
        return {
            "type": self.type,
            "title": self.title,
            "body": self.body,
            "rationale": self.rationale,
            "priority": self.priority,
            "tags": self.tags,
            "metadata": self.metadata,
        }


def _build_context_prompt(
    psyche_state: PsycheState,
    goals: list[Goal],
) -> str:
    """Build the LLM prompt for suggestion generation."""
    goals_text = (
        "\n".join(f"- {g.title}" for g in goals[:5]) if goals else "No active goals"
    )
    distortions = (
        ", ".join(psyche_state.cognitive_distortions)
        if psyche_state.cognitive_distortions
        else "none"
    )
    parts = (
        ", ".join(psyche_state.active_parts[:3])
        if psyche_state.active_parts
        else "none"
    )
    stressors = (
        ", ".join(psyche_state.stressor_tags)
        if psyche_state.stressor_tags
        else "none"
    )

    return (
        f"User's current psychological state:\n"
        f"- Emotional: valence={psyche_state.valence:.2f}, "
        f"arousal={psyche_state.arousal:.2f}, dominance={psyche_state.dominance:.2f}\n"
        f"- Active IFS parts: {parts}\n"
        f"- Dominant need: {psyche_state.dominant_need or 'unknown'}\n"
        f"- Cognitive distortions: {distortions}\n"
        f"- Stressors: {stressors}\n"
        f"- Cognitive load: {psyche_state.cognitive_load:.0%}\n\n"
        f"Active goals:\n{goals_text}\n\n"
        f"Generate 3 proactive suggestions that would genuinely help this person right now.\n"
        f"Each suggestion should be one of these types: "
        f"{', '.join(_SUGGESTION_TYPES)}.\n"
        f"Respond as a JSON array of objects with keys: "
        f'"type", "title", "body", "rationale", "priority" (1-5), "tags" (array).\n'
        f'Example: [{{"type": "health", "title": "Take a break", '
        f'"body": "...", "rationale": "...", "priority": 2, "tags": ["rest"]}}]'
    )


class ProactiveTool(Tool):
    """Chat-accessible proactive suggestions tool.

    Uses the current :class:`~core.psyche.state.PsycheState` and active goals
    to generate contextually relevant suggestions via the LLM.

    Parameters
    ----------
    llm_client:
        LLM client for suggestion generation.  When ``None`` a set of
        heuristic fallback suggestions is returned.
    """

    name = "proactive_suggestions"
    description = (
        "Генерация проактивных предложений на основе текущего состояния пользователя"
    )
    parameters = [
        ToolParameter(
            name="suggestion_types",
            type="string",
            description=(
                "Типы предложений через запятую: "
                "educational, health, productivity, social, creative"
            ),
            required=False,
        ),
        ToolParameter(
            name="limit",
            type="number",
            description="Максимальное количество предложений (по умолчанию 3)",
            required=False,
        ),
    ]

    def __init__(
        self,
        llm_client: LLMClient | None = None,
    ) -> None:
        self._llm = llm_client

    async def execute(self, **kwargs: Any) -> ToolCallResult:
        """Return a stub result; use :meth:`generate_suggestions` for full output."""
        limit = int(kwargs.get("limit", 3))
        data = [
            Suggestion(
                type="productivity",
                title="Просмотрите активные цели",
                body="Найдите одну задачу, которую можно выполнить за 25 минут.",
                rationale="Короткие сессии помогают снизить когнитивную нагрузку.",
                priority=2,
            ).to_dict()
        ]
        return ToolCallResult(
            tool_name=self.name, success=True, data=data[:limit]
        )

    async def generate_suggestions(
        self,
        psyche_state: PsycheState,
        goals: list[Goal],
        limit: int = 3,
        allowed_types: list[str] | None = None,
    ) -> list[Suggestion]:
        """Generate proactive suggestions based on *psyche_state* and *goals*.

        Parameters
        ----------
        psyche_state:
            The user's current psychological state snapshot.
        goals:
            List of the user's active goals.
        limit:
            Maximum number of suggestions to return.
        allowed_types:
            Optional filter; only suggestions of these types will be returned.

        Returns
        -------
        list[Suggestion]
            Up to *limit* suggestions, sorted by priority.
        """
        if self._llm:
            suggestions = await self._llm_suggestions(psyche_state, goals)
        else:
            suggestions = self._heuristic_suggestions(psyche_state, goals)

        if allowed_types:
            suggestions = [s for s in suggestions if s.type in allowed_types]

        suggestions.sort(key=lambda s: s.priority)
        return suggestions[:limit]

    async def _llm_suggestions(
        self,
        psyche_state: PsycheState,
        goals: list[Goal],
    ) -> list[Suggestion]:
        """Generate suggestions using the LLM."""
        prompt = _build_context_prompt(psyche_state, goals)
        try:
            raw = await self._llm.complete(prompt)  # type: ignore[union-attr]
            start = raw.find("[")
            end = raw.rfind("]") + 1
            if start == -1 or end == 0:
                return self._heuristic_suggestions(psyche_state, goals)
            items = json.loads(raw[start:end])
            suggestions: list[Suggestion] = []
            for item in items:
                if not isinstance(item, dict) or not item.get("title"):
                    continue
                suggestions.append(
                    Suggestion(
                        type=item.get("type", "productivity"),
                        title=item["title"],
                        body=item.get("body", ""),
                        rationale=item.get("rationale", ""),
                        priority=int(item.get("priority", 3)),
                        tags=list(item.get("tags", [])),
                    )
                )
            return suggestions
        except Exception:
            logger.warning("ProactiveTool._llm_suggestions failed", exc_info=True)
            return self._heuristic_suggestions(psyche_state, goals)

    def _heuristic_suggestions(
        self,
        psyche_state: PsycheState,
        goals: list[Goal],
    ) -> list[Suggestion]:
        """Generate heuristic fallback suggestions without LLM."""
        suggestions: list[Suggestion] = []

        # Health: low arousal → rest suggestion
        if psyche_state.arousal < -0.3:
            suggestions.append(
                Suggestion(
                    type="health",
                    title="Сделайте перерыв",
                    body="Ваш уровень энергии снижен. Попробуйте 10-минутную прогулку или дыхательное упражнение.",
                    rationale="Низкое возбуждение указывает на усталость.",
                    priority=1,
                    tags=["rest", "energy"],
                )
            )

        # Productivity: negative valence + active goals
        if psyche_state.valence < -0.2 and goals:
            suggestions.append(
                Suggestion(
                    type="productivity",
                    title=f"Маленький шаг к цели: {goals[0].title}",
                    body="Разбейте цель на 3 маленьких действия и сделайте первое прямо сейчас.",
                    rationale="Маленькие победы улучшают настроение.",
                    priority=2,
                    tags=["goals", "momentum"],
                )
            )

        # Educational: cognitive distortions detected
        if psyche_state.cognitive_distortions:
            suggestions.append(
                Suggestion(
                    type="educational",
                    title="Практика когнитивного рефрейминга",
                    body="Запишите мысль, которая вас беспокоит, и найдите 3 альтернативных объяснения ситуации.",
                    rationale=f"Обнаружены когнитивные искажения: {', '.join(psyche_state.cognitive_distortions[:2])}.",
                    priority=2,
                    tags=["cbt", "reframing"],
                )
            )

        # Social: high dominance + no stressors → connect
        if psyche_state.dominance > 0.3 and not psyche_state.stressor_tags:
            suggestions.append(
                Suggestion(
                    type="social",
                    title="Поддержите связь",
                    body="Напишите или позвоните кому-то важному для вас.",
                    rationale="Хорошее состояние — отличное время для общения.",
                    priority=4,
                    tags=["connection", "social"],
                )
            )

        # Creative: fallback
        if not suggestions:
            suggestions.append(
                Suggestion(
                    type="creative",
                    title="Творческая пауза",
                    body="Потратьте 15 минут на что-то приятное: рисование, музыку или письмо.",
                    rationale="Регулярные творческие паузы поддерживают ментальное здоровье.",
                    priority=3,
                    tags=["creativity", "wellbeing"],
                )
            )

        return suggestions
