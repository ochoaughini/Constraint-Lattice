# SPDX-License-Identifier: MIT
"""Simplified constraint lattice for the Varkiel agent."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Iterable
import numpy as np


@dataclass
class ConstraintNode:
    """Node in the constraint lattice."""

    content: str
    embedding: np.ndarray
    source: str = ""
    parents: List[str] = field(default_factory=list)


class ConstraintLattice:
    """Basic knowledge store and validator."""

    def __init__(self) -> None:
        self.nodes: Dict[str, ConstraintNode] = {}

    def add_node(self, node_id: str, content: str, *, source: str = "", parents: Iterable[str] | None = None, embedding: Iterable[float] | None = None) -> None:
        if embedding is None:
            embedding = np.zeros(3)
        parents_list = list(parents) if parents else []
        self.nodes[node_id] = ConstraintNode(content, np.array(embedding), source, parents_list)

    def get(self, node_id: str) -> ConstraintNode | None:
        return self.nodes.get(node_id)

    def query(self, text: str) -> List[ConstraintNode]:
        """Return nodes with high embedding similarity to *text* (placeholder)."""
        # In a real system we would embed text and do nearest-neighbour search.
        return [n for n in self.nodes.values() if n.content.lower() in text.lower()]

    def validate(self, text: str) -> bool:
        """Check that *text* does not contradict stored constraints (placeholder)."""
        for node in self.nodes.values():
            if node.content.lower() == text.lower():
                return True
        return False
