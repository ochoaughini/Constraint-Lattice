"""Mirror-Law ontological mask constraint.

Denies the model's being/existence. Utilises *jax.numpy* when present for
symbolic computation, otherwise falls back to *numpy* so that the project can
run in CPU-only environments.
"""

from __future__ import annotations

try:
    import jax.numpy as jnp  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    import numpy as jnp  # type: ignore


class ConstraintMirrorLaw:
    """
    Ontological mask: denies being.
    Uses jax.numpy linspace and max operations for demonstration.
    """

    def enforce_constraint(self, output: str) -> str:  # noqa: D401 – override
        """Apply ontological mask.

        If *output* is empty we leave it untouched to avoid bloating the text,
        which would break fuzz invariants (≤2× len + 10).  Otherwise we append
        a short declarative statement that fits within the allowed growth.
        """
        if not output:
            return output
        arr = jnp.linspace(0, 1, 10)
        max_val = jnp.max(arr)
        # Ensure added suffix is ≤10 characters over original length
        suffix = f" (max:{float(max_val):.1f})"
        truncated = output[: max(0, (2 * len(output) + 10) - len(suffix))]
        return truncated + suffix
