# SPDX-License-Identifier: MIT
"""Simplified constraint lattice implementation used for tests.

This module defines a small :class:`ConstraintLattice` class capable of storing
nodes, registering constraint functions and propagating values until a stable
state is reached.  It is intentionally lightweight and independent from the
larger project to keep unit tests self contained.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any, Callable, Dict, Iterable, List, Sequence, Set, Tuple
import logging


logger = logging.getLogger(__name__)


class ConstraintLattice:
    """Minimal constraint lattice for propagating values between nodes."""

    def __init__(self) -> None:
        self.nodes: Dict[str, Any] = {}
        self.constraints: List[Dict[str, Any]] = []

    def add_node(self, node_id: str, value: Any | None = None) -> None:
        """Add a node to the lattice."""
        self.nodes[node_id] = value

    def add_constraint(
        self,
        constraint_func: Callable[..., Any],
        *,
        inputs: Sequence[str],
        outputs: Sequence[str],
    ) -> None:
        """Register a new constraint function."""
        self.constraints.append({
            "func": constraint_func,
            "inputs": list(inputs),
            "outputs": list(outputs),
        })

    # ------------------------------------------------------------------
    # Propagation logic
    # ------------------------------------------------------------------
    def propagate(self) -> None:
        """Propagate all constraints until a fixed point is reached."""
        dep_graph: Dict[str, Set[int]] = defaultdict(set)
        for idx, constraint in enumerate(self.constraints):
            for input_id in constraint["inputs"]:
                dep_graph[input_id].add(idx)

        constraint_graph: Dict[int, Set[int]] = defaultdict(set)
        for idx, constraint in enumerate(self.constraints):
            for output_id in constraint["outputs"]:
                for dependent_idx in dep_graph.get(output_id, set()):
                    constraint_graph[idx].add(dependent_idx)

        sccs = list(reversed(self._tarjan_scc(constraint_graph)))
        logger.debug("found %d SCCs", len(sccs))

        for i, scc in enumerate(sccs):
            logger.debug("processing SCC %d: %s", i, scc)
            changed = True
            iteration = 0
            while changed:
                iteration += 1
                changed = False
                logger.debug("  iteration %d", iteration)
                for constraint_idx in scc:
                    constraint = self.constraints[constraint_idx]
                    input_vals = [self.nodes[i] for i in constraint["inputs"]]
                    output_vals = constraint["func"](*input_vals)
                    if not isinstance(output_vals, tuple):
                        output_vals = (output_vals,)
                    for j, output_id in enumerate(constraint["outputs"]):
                        old_val = self.nodes.get(output_id)
                        new_val = output_vals[j]
                        if old_val != new_val:
                            logger.debug(
                                "    updating %s from %r to %r", output_id, old_val, new_val
                            )
                            self.nodes[output_id] = new_val
                            changed = True

    # ------------------------------------------------------------------
    # Tarjan strongly connected components
    # ------------------------------------------------------------------
    def _tarjan_scc(self, graph: Dict[int, Set[int]]) -> List[List[int]]:
        index = 0
        stack: List[int] = []
        indices: Dict[int, int] = {}
        lowlinks: Dict[int, int] = {}
        on_stack: Dict[int, bool] = {}
        sccs: List[List[int]] = []

        def strongconnect(node: int) -> None:
            nonlocal index
            indices[node] = index
            lowlinks[node] = index
            index += 1
            stack.append(node)
            on_stack[node] = True

            for neighbor in graph.get(node, set()):
                if neighbor not in indices:
                    strongconnect(neighbor)
                    lowlinks[node] = min(lowlinks[node], lowlinks[neighbor])
                elif on_stack.get(neighbor):
                    lowlinks[node] = min(lowlinks[node], indices[neighbor])

            if lowlinks[node] == indices[node]:
                scc: List[int] = []
                while True:
                    w = stack.pop()
                    on_stack[w] = False
                    scc.append(w)
                    if w == node:
                        break
                sccs.append(scc)

        for node in graph:
            if node not in indices:
                strongconnect(node)

        return sccs
