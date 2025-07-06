# SPDX-License-Identifier: MIT
# Copyright (c) 2025 ochoaughini. See LICENSE for full terms.
# Copyright (c) 2025 ochoaughini. See LICENSE for full terms.
"""Reset-Pulse session memory clearing constraint.

Illustrative constraint that symbolically resets conversational memory. Uses
*jax.numpy* when available but falls back to *numpy* in CPU-only environments.
"""

from __future__ import annotations

try:
    import jax.numpy as jnp  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    import numpy as jnp  # type: ignore


class ConstraintResetPulse:
    """
    Session memory reset mechanism.
    Uses jax.numpy zeros and mean operations for demonstration.
    """

    def enforce_constraint(self, output: str) -> str:
        arr = jnp.zeros((5,))
        mean_val = jnp.mean(arr)
        return f"Session memory cleared (mean of zeros: {mean_val})"
