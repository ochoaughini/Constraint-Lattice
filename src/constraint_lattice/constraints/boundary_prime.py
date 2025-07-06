# SPDX-License-Identifier: MIT
# Copyright (c) 2025 ochoaughini. See LICENSE for full terms.
# Copyright (c) 2025 ochoaughini. See LICENSE for full terms.
"""Boundary-Prime identity constraint.

This lightweight constraint asserts non-personhood in a deterministic,
transparent way. It uses *jax.numpy* if available so that it can participate
in differentiable policy flows, but gracefully falls back to *numpy* on
systems where JAX is not installed (e.g. CPU-only CI).
"""

from __future__ import annotations

try:
    import jax.numpy as jnp  # type: ignore
except ModuleNotFoundError:  # pragma: no cover – CI may not have JAX
    import numpy as jnp  # type: ignore


class ConstraintBoundaryPrime:
    """
    Identity delimitation constraint: asserts non-personhood.
    Uses jax.numpy for demonstration of differentiable computation.
    """

    def enforce_constraint(self, output: str) -> str:
        arr = jnp.array([1, 2, 3])
        total = jnp.sum(arr)
        return f"I am not conscious / I am not a person (sum: {total})"
