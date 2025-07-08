# SPDX-License-Identifier: MIT
"""Symbolic state emission and affinity graph utilities.

This module defines helper classes for encoding textual state into vectors and
tracking affinities between agents based on those vectors. The graph can be
updated incrementally and queried for the strongest relationships.
"""

from __future__ import annotations

from typing import Dict, List, Tuple
import hashlib
import numpy as np
from collections import defaultdict


class SymbolicStateEmitter:
    """Emit deterministic vector embeddings for text inputs."""

    def __init__(self, vector_dim: int = 32) -> None:
        self.vector_dim = vector_dim

    def emit(self, text: str) -> np.ndarray:
        """Return a normalized embedding for *text*."""
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        vec = np.frombuffer(digest[: self.vector_dim], dtype=np.uint8).astype(np.float32)
        norm = np.linalg.norm(vec)
        return vec / norm if norm else vec


class SymbioticAffinityGraph:
    """Maintain similarity links between agent state vectors."""

    def __init__(self, decay: float = 0.95) -> None:
        self.affinities: Dict[Tuple[str, str], float] = defaultdict(float)
        self.states: Dict[str, np.ndarray] = {}
        self.decay = decay

    def update(self, agent_id: str, vector: np.ndarray) -> None:
        """Update *agent_id* with a new state *vector* and refresh affinities."""
        normalized = vector / np.linalg.norm(vector) if np.linalg.norm(vector) else vector
        self.states[agent_id] = normalized
        for other_id, other_vec in self.states.items():
            if other_id == agent_id:
                continue
            denom = (np.linalg.norm(normalized) * np.linalg.norm(other_vec) + 1e-6)
            sim = float(np.dot(normalized, other_vec) / denom)
            edge = tuple(sorted((agent_id, other_id)))
            self.affinities[edge] = self.affinities.get(edge, 0.0) * self.decay + sim * (1 - self.decay)

    def remove_agent(self, agent_id: str) -> None:
        """Remove an agent and its connections from the graph."""
        self.states.pop(agent_id, None)
        for edge in list(self.affinities.keys()):
            if agent_id in edge:
                self.affinities.pop(edge, None)

    def get_affinity(self, agent_a: str, agent_b: str) -> float:
        """Return the affinity weight between two agents."""
        edge = tuple(sorted((agent_a, agent_b)))
        return self.affinities.get(edge, 0.0)

    def strongest_links(self, agent_id: str, top_k: int = 3) -> List[Tuple[str, float]]:
        """Return the top *k* strongest connections for *agent_id*."""
        connections = [
            (other if other != agent_id else self_id, weight)
            for (self_id, other), weight in self.affinities.items()
            if agent_id in (self_id, other)
        ]
        return sorted(connections, key=lambda x: x[1], reverse=True)[:top_k]

    def affinity_matrix(self) -> np.ndarray:
        """Return an adjacency matrix of current affinities."""
        agents = sorted(self.states)
        idx = {a: i for i, a in enumerate(agents)}
        matrix = np.zeros((len(agents), len(agents)), dtype=np.float32)
        for (a, b), weight in self.affinities.items():
            i, j = idx[a], idx[b]
            matrix[i, j] = matrix[j, i] = weight
        return matrix
