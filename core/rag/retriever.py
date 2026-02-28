"""Graph-aware RAG retriever for SELF-OS.

:class:`GraphRAGRetriever` performs hybrid search over the user's knowledge
graph and enriches each result with its 1-hop neighbours, producing a
structured textual context suitable for LLM prompts.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from core.graph.model import Node
    from core.graph.storage import GraphStorage

__all__ = ["GraphRAGRetriever"]


class GraphRAGRetriever:
    """Retrieve graph nodes relevant to a query and build a textual context.

    Parameters
    ----------
    storage:
        The :class:`~core.graph.storage.GraphStorage` instance to query.
    """

    def __init__(self, storage: "GraphStorage") -> None:
        self.storage = storage

    async def retrieve(
        self,
        user_id: str,
        query: str,
        query_embedding: list[float] | None = None,
        top_k: int = 5,
        alpha: float = 0.7,
    ) -> list[tuple["Node", float]]:
        """Return up to *top_k* nodes most relevant to *query*.

        Uses :meth:`~core.graph.storage.GraphStorage.hybrid_search` which
        combines dense cosine similarity (when *query_embedding* is provided)
        with TF-IDF sparse scoring.
        """
        return await self.storage.hybrid_search(
            user_id=user_id,
            query_text=query,
            query_embedding=query_embedding,
            alpha=alpha,
            top_k=top_k,
        )

    async def build_context(
        self,
        user_id: str,
        query: str,
        query_embedding: list[float] | None = None,
        top_k: int = 5,
    ) -> str:
        """Build a structured textual context string for LLM prompting.

        Each retrieved node is described together with its immediate (1-hop)
        neighbours so that the LLM has relational information, not just
        isolated facts.
        """
        results = await self.retrieve(user_id, query, query_embedding, top_k=top_k)
        if not results:
            return ""

        lines: list[str] = ["=== Retrieved Context ==="]
        for node, score in results:
            node_label = node.name or node.text or node.id
            lines.append(f"[{node.type}] {node_label} (score={score:.3f})")

            # 1-hop neighbours
            out_edges = await self.storage.get_edges_from_node(user_id, node.id)
            in_edges = await self.storage.get_edges_to_node(user_id, node.id)

            neighbour_ids = list({e.target_node_id for e in out_edges} | {e.source_node_id for e in in_edges})
            if neighbour_ids:
                neighbours = await self.storage.get_nodes_by_ids(user_id, neighbour_ids[:5])
                for nb in neighbours:
                    nb_label = nb.name or nb.text or nb.id
                    lines.append(f"  → [{nb.type}] {nb_label}")

        lines.append("=========================")
        return "\n".join(lines)
