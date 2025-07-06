# SPDX-License-Identifier: MIT
# Copyright (c) 2025 ochoaughini. See LICENSE for full terms.
# Copyright (c) 2025 ochoaughini. See LICENSE for full terms.
import os
import types
import sys

import pytest

pytestmark = pytest.mark.skipif(
    os.getenv("CONSTRAINT_LATTICE_USE_JAX", "0") != "1", reason="JAX disabled"
)

import jax
import jax.numpy as jnp

from engine.jax_backend import JAXConstraint


def test_grad_through_constraint():
    """Ensure `jax.grad` works on a wrapped numeric predicate."""

    def squared_sum(x):
        return jnp.sum(x ** 2)

    c = JAXConstraint(squared_sum)

    x = jnp.asarray([1.0, 2.0, 3.0])
    grad = jax.grad(lambda v: c(v))(x)
    assert jnp.allclose(grad, 2 * x)


def test_shape_mismatch_raises():
    """Constraint expecting 1-D vector should fail on 2-D input when batch=False."""

    def first_positive(x):
        assert x.ndim == 1, "Vector expected"
        return jnp.all(x > 0)

    c = JAXConstraint(first_positive)
    bad = jnp.ones((2, 3))
    with pytest.raises(AssertionError):
        _ = c(bad, batch=False)


def test_cpu_fallback(monkeypatch):
    """If JAX flag is off, JAXConstraint should raise at instantiation."""

    monkeypatch.setenv("CONSTRAINT_LATTICE_USE_JAX", "0")
    with pytest.raises(RuntimeError):
        _ = JAXConstraint(lambda x: x)
