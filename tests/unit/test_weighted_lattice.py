import numpy as np
from varkiel.constraint_lattice import ConstraintLattice


def test_linking_and_query():
    cl = ConstraintLattice()
    cl.add_node("a", "hello world")
    cl.add_node("b", "hello there", link_threshold=0.0)
    results = cl.query("hello world")
    assert any(node.content == "hello world" for node in results)
    # ensure edges created
    assert "b" in [n for n, _ in cl.edges.get("a", [])] or "a" in [n for n, _ in cl.edges.get("b", [])]

