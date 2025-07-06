# SPDX-License-Identifier: MIT
"""Tests for cyclic constraint propagation."""

from constraint_lattice_core import ConstraintLattice


def test_cyclic_propagation() -> None:
    lattice = ConstraintLattice()
    lattice.nodes = {"A": 1, "B": 2}

    def constraint1(a: int) -> int:
        return a + 1

    def constraint2(b: int) -> int:
        return b - 1

    lattice.add_constraint(constraint1, inputs=["A"], outputs=["B"])
    lattice.add_constraint(constraint2, inputs=["B"], outputs=["A"])

    lattice.propagate()

    assert lattice.nodes["A"] == 1
    assert lattice.nodes["B"] == 2
