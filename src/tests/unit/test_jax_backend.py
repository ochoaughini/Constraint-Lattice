# SPDX-License-Identifier: MIT
# Copyright (c) 2025 ochoaughini. See LICENSE for full terms.
# Copyright (c) 2025 ochoaughini. See LICENSE for full terms.
import os

import pytest


@pytest.mark.skipif(os.getenv("CONSTRAINT_LATTICE_USE_JAX", "0") != "1", reason="JAX integration disabled")
def test_basic_predicate():
    os.environ["CONSTRAINT_LATTICE_USE_JAX"] = "1"
    import jax.numpy as jnp

    from engine.jax_backend import JAXConstraint

    def all_positive(x):
        return jnp.all(x > 0)

    constraint = JAXConstraint(all_positive)
    assert bool(constraint(jnp.asarray([1, 2, 3]))) is True
    assert bool(constraint(jnp.asarray([-1, 2, 3]))) is False


@pytest.mark.skipif(os.getenv("CONSTRAINT_LATTICE_USE_JAX", "0") != "1", reason="JAX integration disabled")
def test_batching():
    os.environ["CONSTRAINT_LATTICE_USE_JAX"] = "1"
    import jax.numpy as jnp

    from engine.jax_backend import JAXConstraint

    def sum_less_than_ten(x):
        return jnp.sum(x) < 10

    constraint = JAXConstraint(sum_less_than_ten)

    xs = jnp.asarray([[1, 1, 1], [5, 5, 5]])
    result = constraint(xs, batch=True)
    assert result.shape == (2,)
    assert list(map(bool, result)) == [True, False]
