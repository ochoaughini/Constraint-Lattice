import numpy as np
import networkx as nx

from constraint_lattice.engine.graph_ops import (
    NodeData,
    compute_drift,
    detect_cycles,
    collapse_cycles,
)


def test_compute_drift():
    a = np.array([1.0, 0.0])
    b = np.array([0.0, 0.0])
    assert compute_drift(a, b) == np.sqrt(1.0)


def test_detect_cycles_and_collapse():
    g = nx.DiGraph()
    g.add_edge("A", "B")
    g.add_edge("B", "A")
    cycles = detect_cycles(g)
    assert ["A", "B"] in [sorted(c) for c in cycles]

    collapsed = collapse_cycles(g)
    assert collapsed.number_of_nodes() == 1

