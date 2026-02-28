"""RAG generator for SELF-OS.

:class:`RAGGenerator` wraps an LLM client and injects the *retrieved context*
produced by :class:`~core.rag.retriever.GraphRAGRetriever` into the system
prompt so that responses are grounded in the user's personal knowledge graph.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from core.llm_client import LLMClient

logger = logging.getLogger(__name__)

__all__ = ["RAGGenerator"]

_SYSTEM_TEMPLATE = """\
Ты — SELF-OS, персональный AI-ассистент.
Используй следующий контекст из личного графа знаний пользователя при ответе.

{retrieved_context}

Отвечай на том языке, на котором написано сообщение пользователя.
"""


class RAGGenerator:
    """Generate LLM replies augmented with retrieved graph context.

    Parameters
    ----------
    llm_client:
        Any object that exposes ``generate_live_reply`` compatible with
        :class:`~core.llm_client.LLMClient`.
    """

    def __init__(self, llm_client: "LLMClient") -> None:
        self.llm_client = llm_client

    async def generate(
        self,
        user_text: str,
        retrieved_context: str,
        intent: str = "UNKNOWN",
        mood_context: dict | None = None,
        parts_context: list[dict] | None = None,
    ) -> str:
        """Generate a reply grounded in *retrieved_context*.

        Parameters
        ----------
        user_text:
            The original user message.
        retrieved_context:
            Textual context built by
            :meth:`~core.rag.retriever.GraphRAGRetriever.build_context`.
        intent:
            Classified intent of the user message.
        mood_context:
            Optional mood snapshot dict.
        parts_context:
            Optional list of IFS-part dicts.

        Returns
        -------
        str
            The generated reply, or an empty string on failure.
        """
        system_prompt = _SYSTEM_TEMPLATE.format(
            retrieved_context=retrieved_context or "(контекст отсутствует)"
        )
        try:
            graph_context: dict = {"rag_system_prompt": system_prompt}
            return await self.llm_client.generate_live_reply(
                user_text=user_text,
                intent=intent,
                mood_context=mood_context,
                parts_context=parts_context,
                graph_context=graph_context,
            )
        except Exception as exc:
            logger.warning("RAGGenerator.generate failed: %s", exc)
            return ""
