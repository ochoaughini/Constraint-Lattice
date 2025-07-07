# SPDX-License-Identifier: MIT
"""Utility functions for advanced constraint graph operations.

This module implements helpers referenced in the advanced documentation:
- drift metric computation
- cycle detection and collapsing
- node metadata container
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import networkx as nx


@dataclass
class NodeData:
    """Metadata stored on each node in the constraint graph."""

    vector: np.ndarray
    active: bool = False
    resonance_score: float = 0.0


def compute_drift(current: np.ndarray, reference: np.ndarray) -> float:
    """Return the L2 drift between two embeddings."""
    return float(np.linalg.norm(current - reference))


def detect_cycles(graph: nx.DiGraph) -> List[List[str]]:
    """Return a list of cycles discovered in *graph*.

    Each cycle is represented as a list of node IDs. Single nodes are
    omitted from the result.
    """
    components = list(nx.strongly_connected_components(graph))
    return [list(c) for c in components if len(c) > 1]


def collapse_cycles(graph: nx.DiGraph) -> nx.DiGraph:
    """Collapse strongly connected components into single nodes."""
    sccs = list(nx.strongly_connected_components(graph))
    if all(len(c) == 1 for c in sccs):
        return graph.copy()

    mapping: Dict[str, str] = {}
    collapsed = nx.DiGraph()

    for idx, comp in enumerate(sccs):
        if len(comp) == 1:
            node = next(iter(comp))
            mapping[node] = node
            if node not in collapsed:
                collapsed.add_node(node, **graph.nodes[node])
            continue
        name = f"cycle_{idx}"
        for n in comp:
            mapping[n] = name
        collapsed.add_node(name)

    for u, v, data in graph.edges(data=True):
        src = mapping[u]
        dst = mapping[v]
        if src != dst:
            collapsed.add_edge(src, dst, **data)

    return collapsed
