# SPDX-License-Identifier: MIT
import importlib.util
from pathlib import Path

import numpy as np
import networkx as nx

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location(
    "autolearning", ROOT / "src/constraint_lattice/engine/autolearning.py"
)
autolearning = importlib.util.module_from_spec(spec)
spec.loader.exec_module(autolearning)

apply_containment_constraints = autolearning.apply_containment_constraints
recursive_autolearning_orchestrator = autolearning.recursive_autolearning_orchestrator


def test_apply_containment_constraints_basic():
    g = nx.DiGraph()
    g.add_edge("A", "B")
    state = {"A": np.array([1.0, 0.0]), "B": np.array([0.0, 1.0])}
    feedback = np.array([1.0, 0.0])
    updated = apply_containment_constraints(
        state,
        feedback,
        g,
        containment_threshold=0.1,
        amplification_threshold=0.9,
    )
    assert updated.nodes["A"].get("active") is True
    assert updated.nodes["B"].get("active") is False


def test_recursive_autolearning_orchestrator_updates():
    g = nx.DiGraph()
    g.add_node("A", vector=np.array([1.0, 0.0]))
    memory = np.array([0.0, 1.0])
    status, new_topology = recursive_autolearning_orchestrator(
        g, memory, epoch=1, drift_threshold=0.5
    )
    assert status == "updated"
    assert "strategy" in new_topology.graph

