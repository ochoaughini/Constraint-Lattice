# SPDX-License-Identifier: MIT
"""Simplified constraint lattice for the Varkiel agent."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Iterable, Tuple
import hashlib
import numpy as np


@dataclass
class ConstraintNode:
    """Node in the constraint lattice."""

    content: str
    embedding: np.ndarray
    source: str = ""
    parents: List[str] = field(default_factory=list)


class ConstraintLattice:
    """Basic knowledge store and validator with weighted edges."""

    def __init__(self) -> None:
        self.nodes: Dict[str, ConstraintNode] = {}
        self.edges: Dict[str, List[Tuple[str, float]]] = {}

    def _sim(self, a: np.ndarray, b: np.ndarray) -> float:
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-6))

    def _default_embed(self, text: str) -> np.ndarray:
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        vec = np.frombuffer(digest[:32], dtype=np.uint8).astype(np.float32)
        norm = np.linalg.norm(vec)
        return vec / norm if norm else vec

    def add_node(
        self,
        node_id: str,
        content: str,
        *,
        source: str = "",
        parents: Iterable[str] | None = None,
        embedding: Iterable[float] | None = None,
        link_threshold: float = 0.8,
    ) -> None:
        if embedding is None:
            embedding = self._default_embed(content)
        parents_list = list(parents) if parents else []
        node_vec = np.array(embedding, dtype=np.float32)
        self.nodes[node_id] = ConstraintNode(content, node_vec, source, parents_list)
        self.edges[node_id] = []
        # link to existing nodes
        for other_id, other in self.nodes.items():
            if other_id == node_id:
                continue
            weight = self._sim(node_vec, other.embedding)
            if weight >= link_threshold:
                self.edges[node_id].append((other_id, weight))

    def get(self, node_id: str) -> ConstraintNode | None:
        return self.nodes.get(node_id)

    def query(self, text: str, *, hops: int = 1, threshold: float = 0.8) -> List[ConstraintNode]:
        """Return nodes with high embedding similarity to *text* using optional multi-hop."""
        qvec = np.array(self._default_embed(text))
        results: List[Tuple[float, ConstraintNode]] = []
        for nid, node in self.nodes.items():
            sim = self._sim(qvec, node.embedding)
            if sim >= threshold:
                results.append((sim, node))
        if results or hops <= 0:
            return [n for _, n in sorted(results, reverse=True)]

        # multi-hop search
        for nid, edges in self.edges.items():
            for neigh_id, weight in edges:
                node = self.nodes[neigh_id]
                sim = self._sim(qvec, node.embedding) * weight
                if sim >= threshold * 0.9:
                    results.append((sim, node))
        return [n for _, n in sorted(results, reverse=True)]

    def validate(self, text: str) -> bool:
        """Check that *text* does not contradict stored constraints (placeholder)."""
        for node in self.nodes.values():
            if node.content.lower() == text.lower():
                return True
        return False
