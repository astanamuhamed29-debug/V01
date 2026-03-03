"""Bridge between the OODA pipeline and the NeuroCore engine.

Translates existing ``Node`` / ``Edge`` objects produced by the ORIENT
stage into NeuroCore ``Neuron`` / ``Synapse`` records, and exposes the
resulting ``BrainState`` back to the pipeline for richer reply context.

Usage inside the DECIDE stage::

    bridge = NeuroBridge(neuro_core)
    brain_state = await bridge.mirror(user_id, created_nodes, created_edges)
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from core.neuro.schema import BrainState

if TYPE_CHECKING:
    from core.graph.model import Edge, Node
    from core.neuro.engine import NeuroCore

logger = logging.getLogger(__name__)

# Node.type (uppercase) → neuron_type (lowercase) mapping.
_TYPE_MAP: dict[str, str] = {
    "EMOTION": "emotion",
    "PART": "part",
    "BELIEF": "belief",
    "NEED": "need",
    "VALUE": "value",
    "THOUGHT": "thought",
    "NOTE": "memory",
    "EVENT": "event",
    "SOMA": "soma",
    "INSIGHT": "insight",
    "PROJECT": "event",
    "TASK": "event",
    "PERSON": "memory",
}


class NeuroBridge:
    """Thin adapter that mirrors OODA pipeline artifacts into NeuroCore."""

    def __init__(self, neuro_core: NeuroCore) -> None:
        self._nc = neuro_core

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def mirror(
        self,
        user_id: str,
        nodes: list[Node],
        edges: list[Edge],
    ) -> BrainState:
        """Mirror pipeline nodes/edges into NeuroCore and return brain state.

        1. Each ``Node`` is activated as a ``Neuron``.
        2. Each ``Edge`` is connected as a ``Synapse``.
        3. Co-activated neurons get Hebbian strengthening.
        4. A ``BrainState`` snapshot is taken and returned.
        """
        neuron_ids: list[str] = []

        for node in nodes:
            neuron = await self._activate_from_node(user_id, node)
            if neuron:
                neuron_ids.append(neuron.id)

        for edge in edges:
            await self._connect_from_edge(user_id, edge)

        # Hebbian strengthening for co-activated neurons
        if len(neuron_ids) >= 2:
            await self._nc.hebbian_strengthen(user_id, neuron_ids)

        # Capture and persist brain state snapshot
        state = await self._nc.snapshot_state(user_id)
        return state

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _activate_from_node(self, user_id: str, node: Node):
        """Activate a NeuroCore neuron from a pipeline Node."""
        neuron_type = _TYPE_MAP.get(node.type, "memory")
        content = node.text or node.name or ""
        if not content:
            return None

        meta = dict(node.metadata) if node.metadata else {}
        # Carry over the original graph node id for traceability
        meta["graph_node_id"] = node.id

        valence = float(meta.pop("pad_v", meta.pop("valence", 0.0)))
        arousal = float(meta.pop("pad_a", meta.pop("arousal", 0.0)))
        dominance = float(meta.pop("pad_d", meta.pop("dominance", 0.5)))

        try:
            neuron = await self._nc.activate(
                user_id=user_id,
                neuron_type=neuron_type,
                content=content,
                neuron_id=node.id,
                valence=valence,
                arousal=arousal,
                dominance=dominance,
                metadata=meta,
            )
            return neuron
        except Exception as exc:  # pragma: no cover
            logger.warning("NeuroBridge: failed to activate neuron for node %s: %s", node.id, exc)
            return None

    async def _connect_from_edge(self, user_id: str, edge: Edge):
        """Create a NeuroCore synapse from a pipeline Edge."""
        try:
            await self._nc.connect(
                user_id=user_id,
                source_id=edge.source_node_id,
                target_id=edge.target_node_id,
                relation=edge.relation,
                metadata=dict(edge.metadata) if edge.metadata else {},
            )
        except Exception as exc:  # pragma: no cover
            logger.warning("NeuroBridge: failed to create synapse for edge %s: %s", edge.id, exc)
