from __future__ import annotations

"""JAX execution utilities for Constraint-Lattice.

This module is **fully optional** and only imported when the environment
variable ``CONSTRAINT_LATTICE_USE_JAX`` is set *and* the optional
``constraint-lattice[jax]`` extra is installed.  The goal is to offer a thin,
well-typed wrapper that turns a pure Python function into a compiled, batched,
accelerator-friendly closure while remaining *transparent* to the rest of the
codebase.

Usage example
-------------

>>> import jax.numpy as jnp
>>> from engine.jax_backend import JAXConstraint
>>>
>>> def _safe_sum(x: jnp.ndarray) -> bool:  # A toy predicate
...     return jnp.all(jnp.sum(x) < 10)
>>>
>>> safe_sum = JAXConstraint(_safe_sum)
>>> safe_sum(jnp.asarray([1, 2, 3]))
Array(True, dtype=bool)

The returned object is a ``jax.Array`` so it can be consumed by other JAX
kernels or converted to a Python ``bool`` via ``bool()`` if required.
"""

from dataclasses import dataclass
from functools import partial
from typing import Any, Callable, Tuple
import os

# ----------------------------------------------------------------------------
# Optional JAX import.
# ----------------------------------------------------------------------------

_JAX_ENABLED = os.getenv("CONSTRAINT_LATTICE_USE_JAX", "0") == "1"

try:
    if _JAX_ENABLED:
        import jax  # type: ignore
        import jax.numpy as jnp  # type: ignore
    else:
        raise ImportError  # Skip import if feature-flag is off
except ImportError:  # pragma: no cover – gracefully degraded runtime
    # Provide *very* thin shims so that static type-checking still works even
    # when the extra dependency is absent.
    class _JaxStubModule:  # pylint: disable=too-few-public-methods
        def __getattr__(self, item: str):  # noqa: D401 – intentional simplicity
            raise RuntimeError(
                "JAX support is disabled. Install optional deps with\n"
                "    pip install 'constraint-lattice[jax]'\n"
                "and set the env var CONSTRAINT_LATTICE_USE_JAX=1 to enable."
            )

    jax = _JaxStubModule()  # type: ignore
    jnp = _JaxStubModule()  # type: ignore

# ----------------------------------------------------------------------------
# Public API
# ----------------------------------------------------------------------------

BatchedArray = "jax.Array"  # Forward declaration for type hints


@dataclass
class JAXConstraint:
    """Wrap a *pure* Python predicate into a compiled JAX closure.

    Parameters
    ----------
    fn:
        A side-effect-free callable that returns a JAX-compatible tensor (e.g.
        ``bool`` or numeric) given one or more input tensors.
    static_argnums:
        Positional argument indices that should be treated as *static* (compile
       -time constants) by XLA. Mimics the ``static_argnums`` argument of
        :pymeth:`jax.jit`.
    rng_seed:
        Seed for the per-instance PRNG.  This is useful when the wrapped
        function relies on :pydata:`jax.random` but you still want determinism
        across runs.
    """

    fn: Callable[..., Any]
    static_argnums: Tuple[int, ...] = ()
    rng_seed: int = 0

    def __post_init__(self):
        if not _JAX_ENABLED:
            raise RuntimeError(
                "JAXConstraint instantiated but JAX runtime is not enabled.\n"
                "Install extras and export CONSTRAINT_LATTICE_USE_JAX=1."
            )
        # Compile once.  Subsequent calls will hit the cached version.
        self._compiled = jax.jit(self.fn, static_argnums=self.static_argnums)
        # Pre-allocate a PRNG key for deterministic randomness inside the
        # predicate (if any). Child keys are generated on each `__call__`.
        self._rng = jax.random.PRNGKey(self.rng_seed)

    # ---------------------------------------------------------------------
    # Functional interface
    # ---------------------------------------------------------------------

    from typing import Optional

    def __call__(self, *args, batch: Optional[bool] = None, **kwargs):  # noqa: D401
        """Invoke the compiled predicate.

        If *batch* is ``True`` we apply :pyfunc:`jax.vmap` across the *first*
        axis of **every** positional argument, returning a vectorised result.
        If *batch* is ``False`` we execute the scalar predicate directly.
        When *batch* is ``None`` (default) we auto-detect batching based on the
        dimensionality of the first tensor argument.
        """

        if batch is None and args and hasattr(args[0], "ndim"):
            batch = args[0].ndim > 1  # type: ignore[attr-defined]

        if batch:
            vectorised = jax.vmap(self._compiled)
            return vectorised(*args, **kwargs)

        return self._compiled(*args, **kwargs)

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_bool_fn(
        cls,
        bool_fn: Callable[[Any], bool],
        *,
        static_argnums: Tuple[int, ...] = (),
        rng_seed: int = 0,
    ) -> "JAXConstraint":
        """Utility to wrap a *Python* Boolean function in one call."""

        def _wrapped(*a):  # type: ignore[override]
            return bool_fn(*a)

        return cls(_wrapped, static_argnums=static_argnums, rng_seed=rng_seed)

    # ------------------------------------------------------------------
    # Introspection – useful for the audit log.
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:  # noqa: D401 – property makes more sense here
        """Return dotted path of the original function for traceability."""

        mod = self.fn.__module__
        qual = getattr(self.fn, "__qualname__", self.fn.__name__)
        return f"{mod}.{qual}"

    # ------------------------------------------------------------------
    # String representation – helps when dumping audit traces.
    # ------------------------------------------------------------------

    def __repr__(self):  # noqa: D401 – standard
        return f"<JAXConstraint {self.name}>"
