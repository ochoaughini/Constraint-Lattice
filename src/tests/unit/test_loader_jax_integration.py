import os
import sys
import types
import textwrap
from pathlib import Path

import pytest

# Skip tests if JAX feature flag is off.
pytestmark = pytest.mark.skipif(
    os.getenv("CONSTRAINT_LATTICE_USE_JAX", "0") != "1", reason="JAX integration disabled"
)


@pytest.fixture(scope="module")
def _enable_jax_env():
    """Ensure JAX flag is set for the duration of this module."""

    old_val = os.getenv("CONSTRAINT_LATTICE_USE_JAX")
    os.environ["CONSTRAINT_LATTICE_USE_JAX"] = "1"
    yield
    # Restore environment to previous state
    if old_val is not None:
        os.environ["CONSTRAINT_LATTICE_USE_JAX"] = old_val
    else:
        del os.environ["CONSTRAINT_LATTICE_USE_JAX"]


@pytest.fixture(scope="module")
def dummy_constraints_module(_enable_jax_env):
    """Create an in-memory module with a JAX-friendly predicate function."""

    import jax.numpy as jnp

    mod = types.ModuleType("dummy_constraints")

    def all_positive(x):
        return jnp.all(x > 0)

    mod.all_positive = all_positive  # type: ignore[attr-defined]
    sys.modules[mod.__name__] = mod
    return mod.__name__


def _write_yaml(tmp_path: Path, content: str) -> Path:
    yaml_path = tmp_path / "constraints.yaml"
    yaml_path.write_text(textwrap.dedent(content))
    return yaml_path


def test_function_constraint_wrapped(tmp_path, dummy_constraints_module):
    """Loader should wrap function-style constraints when engine: jax is set."""

    yaml_path = _write_yaml(
        tmp_path,
        f"""
        profiles:
          default:
            - {{ class: all_positive, engine: jax }}
        """,
    )

    from engine.loader import load_constraints_from_yaml

    constraints = load_constraints_from_yaml(
        str(yaml_path), "default", [dummy_constraints_module]
    )
    assert len(constraints) == 1

    # The loader is expected to return an *instance* whose call/evaluate returns a JAX bool.
    c = constraints[0]
    import jax.numpy as jnp

    result = c(jnp.asarray([1, 2, 3])) if callable(c) else c.enforce_constraint(jnp.asarray([1, 2, 3]))  # type: ignore[arg-type]
    assert bool(result) is True


def test_class_constraint_fallback(tmp_path):
    """Class-based constraints still load even if wrapping fails."""

    # Use an existing constraint that returns a *string*, incompatible with JAX compilation.
    yaml_path = _write_yaml(
        tmp_path,
        """
        profiles:
          default:
            - { class: ConstraintBoundaryPrime, engine: jax }
        """,
    )

    from engine.loader import load_constraints_from_yaml

    constraints = load_constraints_from_yaml(
        str(yaml_path), "default", ["constraints.boundary_prime"]
    )
    assert len(constraints) == 1

    inst = constraints[0]
    out = inst.enforce_constraint("irrelevant")
    assert isinstance(out, str) and "not conscious" in out.lower()
