import numpy as np
from constraint_lattice.engine.symbiotic_state import (
    SymbolicStateEmitter,
    SymbioticAffinityGraph,
)


def test_emitter_returns_normalized_vector():
    emitter = SymbolicStateEmitter(vector_dim=16)
    vec = emitter.emit("hello")
    norm = np.linalg.norm(vec)
    assert vec.shape == (16,)
    assert np.isclose(norm, 1.0)


def test_affinity_graph_updates_and_queries():
    emitter = SymbolicStateEmitter(vector_dim=8)
    graph = SymbioticAffinityGraph(decay=0.0)

    vec_a = emitter.emit("agent_a")
    vec_b = emitter.emit("agent_b")

    graph.update("a", vec_a)
    graph.update("b", vec_b)

    affinity = graph.get_affinity("a", "b")
    assert affinity > 0

    strongest = graph.strongest_links("a", top_k=1)
    assert strongest[0][0] == "b"

    graph.remove_agent("a")
    assert graph.get_affinity("a", "b") == 0
