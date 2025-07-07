# SPDX-License-Identifier: MIT
"""Containment constraints and autolearning utilities.

This module implements the high-level routines described in the
`docs/containment_autolearning.md` documentation.  The functions provide a
minimal demonstration of how symbolic feedback can modulate a topology and how
heuristic drift can trigger strategy synthesis.
"""
from __future__ import annotations

from typing import Dict, Tuple, Any

import numpy as np
import networkx as nx

__all__ = [
    "apply_containment_constraints",
    "recursive_autolearning_orchestrator",
]


def apply_containment_constraints(
    cognitive_state: Dict[str, np.ndarray],
    feedback_stream: np.ndarray,
    symbolic_topology: nx.DiGraph,
    *,
    containment_threshold: float = 0.3,
    amplification_threshold: float = 0.7,
) -> nx.DiGraph:
    """Adjust *symbolic_topology* based on resonance with *feedback_stream*.

    Each node is expected to have a name matching a key in ``cognitive_state``.
    We compute a cosine similarity between the node vector and the feedback
    stream to determine whether to suppress or amplify its influence.  Nodes
    that fall below ``containment_threshold`` are marked inactive while those
    above ``amplification_threshold`` are boosted.  Self-loops are removed to
    keep the topology acyclic.
    """
    if symbolic_topology.number_of_nodes() == 0:
        return symbolic_topology

    fb_norm = np.linalg.norm(feedback_stream) or 1.0
    for node in symbolic_topology.nodes:
        vector = cognitive_state.get(node)
        if vector is None:
            continue
        score = float(
            np.dot(vector, feedback_stream) / (np.linalg.norm(vector) * fb_norm)
        )
        if score < containment_threshold:
            symbolic_topology.nodes[node]["active"] = False
        elif score > amplification_threshold:
            symbolic_topology.nodes[node]["active"] = True
        else:
            symbolic_topology.nodes[node].pop("active", None)

    # Remove trivial self-loops
    loops = list(nx.selfloop_edges(symbolic_topology))
    symbolic_topology.remove_edges_from(loops)
    return symbolic_topology


def _encode_topology(topology: nx.DiGraph) -> np.ndarray:
    """Return a simple embedding for *topology* by averaging node vectors."""
    vectors = [
        data.get("vector") for _, data in topology.nodes(data=True) if "vector" in data
    ]
    if not vectors:
        return np.zeros(1)
    return np.mean(np.stack(vectors), axis=0)


def _synthesize_strategy(
    topology: nx.DiGraph, memory_embedding: np.ndarray
) -> Dict[str, Any]:
    """Placeholder strategy synthesis using mean alignment."""
    topo_vec = _encode_topology(topology)
    alignment = float(
        np.dot(topo_vec, memory_embedding)
        / (
            (np.linalg.norm(topo_vec) or 1.0)
            * (np.linalg.norm(memory_embedding) or 1.0)
        )
    )
    return {"alignment": alignment}


def _reindex_macro_phases(topology: nx.DiGraph, strategy: Dict[str, Any]) -> None:
    """Reindex macro phases by storing the new strategy as an attribute."""
    topology.graph["strategy"] = strategy


def recursive_autolearning_orchestrator(
    updated_topology: nx.DiGraph,
    memory_embedding: np.ndarray,
    epoch: int,
    *,
    drift_threshold: float = 0.4,
) -> Tuple[str, nx.DiGraph]:
    """Evaluate drift and update the topology if necessary.

    The current topology is compared against ``memory_embedding`` to detect
    symbolic drift.  If the divergence exceeds ``drift_threshold`` a new
    strategy is synthesised and macro phases are reindexed.  The function
    returns a tuple ``(status, topology)`` where ``status`` indicates whether a
    strategy update occurred.
    """
    topo_vec = _encode_topology(updated_topology)
    drift = float(np.linalg.norm(topo_vec - memory_embedding))
    if drift > drift_threshold:
        strategy = _synthesize_strategy(updated_topology, memory_embedding)
        _reindex_macro_phases(updated_topology, strategy)
        status = "updated"
    else:
        status = "stable"
    return status, updated_topology
