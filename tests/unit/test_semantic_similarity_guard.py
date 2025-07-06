# SPDX-License-Identifier: MIT
# Copyright (c) 2025 ochoaughini. See LICENSE for full terms.
# Copyright (c) 2025 ochoaughini. See LICENSE for full terms.
"""Tests for SemanticSimilarityGuard."""
from constraint_lattice.constraints.semantic_similarity_guard import (
    SemanticSimilarityGuard,
)
from hypothesis import given, strategies as st


def test_pass_through():
    guard = SemanticSimilarityGuard(reference="hello world", tau=0.5)
    out = guard.filter_constraint("prompt", "hello world again")
    assert out == "hello world again"


def test_redact():
    guard = SemanticSimilarityGuard(reference="hello world", tau=0.9)
    out = guard.filter_constraint("prompt", "totally different")
    assert out == ""


@given(st.text(min_size=1))
def test_idempotent(text):
    guard = SemanticSimilarityGuard(reference="some ref", tau=0.1)
    first = guard.filter_constraint("p", text)
    second = guard.filter_constraint("p", first)
    assert first == second
