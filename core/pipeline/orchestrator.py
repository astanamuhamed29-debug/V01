"""Multi-Agent Orchestration layer for SELF-OS.

Implements a LangGraph-style agent graph where each pipeline step can be
routed to a specialised agent. Each agent exposes a single async interface::

    async def run(context: AgentContext) -> AgentResult

The :class:`AgentOrchestrator` selects the right chain of agents based on
intent and graph/mood/parts context, with a fallback strategy when any
single agent fails.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data-transfer objects
# ---------------------------------------------------------------------------


@dataclass
class AgentContext:
    """Shared context passed between agents in a single pipeline turn."""

    user_id: str
    text: str
    intent: str
    graph_context: dict[str, Any] = field(default_factory=dict)
    mood_context: dict[str, Any] = field(default_factory=dict)
    parts_context: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class AgentResult:
    """Result produced by a single agent execution."""

    nodes: list[dict[str, Any]] = field(default_factory=list)
    edges: list[dict[str, Any]] = field(default_factory=list)
    reply_fragment: str = ""
    next_agent: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------


class BaseAgent:
    """Abstract base for all SELF-OS agents."""

    name: str = "base"

    async def run(self, context: AgentContext) -> AgentResult:  # pragma: no cover
        """Execute agent logic. Must be overridden by subclasses."""
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Specialised agents
# ---------------------------------------------------------------------------


class SemanticExtractorAgent(BaseAgent):
    """Extracts semantic nodes (projects, tasks, beliefs, notes) from text."""

    name = "semantic_extractor"

    async def run(self, context: AgentContext) -> AgentResult:
        """Extract semantic structures from *context.text*."""
        nodes: list[dict[str, Any]] = []
        reply_fragment = ""

        text_lower = context.text.lower()

        if any(kw in text_lower for kw in ("проект", "project", "разработ", "создать")):
            nodes.append({"type": "PROJECT", "source": "semantic_agent"})

        if any(kw in text_lower for kw in ("задача", "task", "надо", "нужно", "сделать")):
            nodes.append({"type": "TASK", "source": "semantic_agent"})

        if any(kw in text_lower for kw in ("верю", "убежден", "я думаю", "кажется")):
            nodes.append({"type": "BELIEF", "source": "semantic_agent"})

        if nodes:
            reply_fragment = f"Нашёл {len(nodes)} семантических элемент(а/ов)."

        return AgentResult(nodes=nodes, reply_fragment=reply_fragment)


class EmotionAnalysisAgent(BaseAgent):
    """Analyses emotional content and builds mood context."""

    name = "emotion_analysis"

    async def run(self, context: AgentContext) -> AgentResult:
        """Identify emotions expressed in *context.text*."""
        nodes: list[dict[str, Any]] = []
        text_lower = context.text.lower()

        emotion_keywords: dict[str, list[str]] = {
            "тревога": ["тревог", "тревож", "беспокой", "волну", "страх"],
            "радость": ["радост", "счастл", "рад ", "доволен"],
            "грусть": ["грустн", "печал", "тоскл", "уныл"],
            "злость": ["злост", "злюсь", "раздраж", "бешен"],
        }

        for label, keywords in emotion_keywords.items():
            if any(kw in text_lower for kw in keywords):
                nodes.append({"type": "EMOTION", "label": label, "source": "emotion_agent"})

        return AgentResult(nodes=nodes)


class PartsDetectorAgent(BaseAgent):
    """Detects IFS-model Parts mentioned in the message."""

    name = "parts_detector"

    async def run(self, context: AgentContext) -> AgentResult:
        """Identify Internal Family Systems parts in *context.text*."""
        nodes: list[dict[str, Any]] = []
        text_lower = context.text.lower()

        part_signals: dict[str, list[str]] = {
            "MANAGER": ["контролирую", "слежу", "организую", "планирую"],
            "FIREFIGHTER": ["отвлекаюсь", "убегаю", "спасаюсь", "игнорирую"],
            "EXILE": ["боюсь", "стыжусь", "не могу", "одинок"],
        }

        for subtype, keywords in part_signals.items():
            if any(kw in text_lower for kw in keywords):
                nodes.append({"type": "PART", "subtype": subtype, "source": "parts_agent"})

        return AgentResult(nodes=nodes)


class ConflictResolverAgent(BaseAgent):
    """Detects and attempts to resolve conflicts between Parts and Values."""

    name = "conflict_resolver"

    async def run(self, context: AgentContext) -> AgentResult:
        """Check graph context for existing conflicts and suggest resolution."""
        edges: list[dict[str, Any]] = []
        reply_fragment = ""

        has_conflict = context.graph_context.get("session_conflict", False)
        if has_conflict:
            reply_fragment = (
                "Замечаю внутренний конфликт между частями. "
                "Давай разберёмся, что важнее прямо сейчас."
            )
            edges.append({"relation": "CONFLICTS_WITH", "source": "conflict_agent"})

        return AgentResult(edges=edges, reply_fragment=reply_fragment)


class InsightGeneratorAgent(BaseAgent):
    """Generates reflective insights based on the accumulated context."""

    name = "insight_generator"

    async def run(self, context: AgentContext) -> AgentResult:
        """Produce a reflective insight fragment from mood and parts context."""
        reply_fragment = ""

        mood = context.mood_context
        if mood:
            label = mood.get("dominant_label", "")
            if label:
                reply_fragment = f"Похоже, сейчас доминирует состояние «{label}»."

        parts = context.parts_context
        if parts:
            names = [p.get("name") or p.get("key") or "часть" for p in parts[:2]]
            parts_str = " и ".join(names)
            suffix = f" Замечаю активные части: {parts_str}."
            reply_fragment = (reply_fragment + suffix).strip()

        return AgentResult(reply_fragment=reply_fragment)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_AGENTS: dict[str, BaseAgent] = {
    agent.name: agent
    for agent in [
        SemanticExtractorAgent(),
        EmotionAnalysisAgent(),
        PartsDetectorAgent(),
        ConflictResolverAgent(),
        InsightGeneratorAgent(),
    ]
}

# Intent → preferred agent chain
_INTENT_CHAINS: dict[str, list[str]] = {
    "FEELING_REPORT": ["emotion_analysis", "parts_detector", "conflict_resolver", "insight_generator"],
    "EVENT_REPORT": ["semantic_extractor", "emotion_analysis", "insight_generator"],
    "META": ["semantic_extractor", "conflict_resolver", "insight_generator"],
    "TASK_REPORT": ["semantic_extractor", "insight_generator"],
    "REFLECTION": ["emotion_analysis", "parts_detector", "insight_generator"],
    "UNKNOWN": ["semantic_extractor", "emotion_analysis"],
}

_DEFAULT_CHAIN: list[str] = ["semantic_extractor", "emotion_analysis", "insight_generator"]


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


class AgentOrchestrator:
    """Routes messages through a conditional chain of specialised agents.

    Usage::

        orchestrator = AgentOrchestrator()
        result = await orchestrator.run(context)
    """

    def __init__(self, agents: dict[str, BaseAgent] | None = None) -> None:
        self._agents: dict[str, BaseAgent] = agents if agents is not None else dict(_AGENTS)

    def _get_chain(self, intent: str) -> list[str]:
        """Return the agent chain names for *intent*."""
        return _INTENT_CHAINS.get(intent, _DEFAULT_CHAIN)

    async def run(self, context: AgentContext) -> AgentResult:
        """Execute the agent chain for *context.intent* and merge results.

        Falls back to the next agent in the chain if any single agent raises
        an exception.
        """
        chain = self._get_chain(context.intent)
        merged = AgentResult()

        current_context = context
        for agent_name in chain:
            agent = self._agents.get(agent_name)
            if agent is None:
                logger.warning("Agent %r not found, skipping", agent_name)
                continue
            try:
                result = await agent.run(current_context)
                merged.nodes.extend(result.nodes)
                merged.edges.extend(result.edges)
                if result.reply_fragment:
                    merged.reply_fragment = (
                        (merged.reply_fragment + " " + result.reply_fragment).strip()
                    )
                merged.metadata.update(result.metadata)
                if result.next_agent and result.next_agent in self._agents:
                    logger.debug("Agent %r requested routing to %r", agent_name, result.next_agent)
            except Exception as exc:
                logger.warning("Agent %r failed: %s — continuing with fallback", agent_name, exc)

        return merged
