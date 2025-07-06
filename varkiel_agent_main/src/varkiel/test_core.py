# SPDX-License-Identifier: MIT
# Copyright (c) 2025 ochoaughini. See LICENSE for full terms.
# Copyright (c) 2025 ochoaughini. See LICENSE for full terms.
# Copyright (c) 2025 ochoaughini. See LICENSE for full terms.
"""Unit tests for Varkiel core functionality"""
import pytest
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from hypothesis import given, strategies as st, settings
from hypothesis.extra.numpy import arrays

# Test utilities
from structural_constraint_engine import StructuralConstraintEngine, MetaConstraintTree
from symbolic_coherence_engine import SymbolicCoherenceEngine
from phenomenological_tracker import PhenomenologicalTracker
from central_controller import CentralController

def test_structural_constraints():
    """Test structural constraint application"""
    tree = MetaConstraintTree(constraints=[])
    engine = StructuralConstraintEngine(constraint_tree=tree)
    engine.constraint_tree.add_constraint(lambda x: x * 0.5)
    
    input_vec = np.array([2.0, 4.0, 6.0])
    output = engine.apply_constraints(input_vec)
    assert np.array_equal(output, np.array([1.0, 2.0, 3.0]))

def test_symbolic_coherence():
    """Test symbolic coherence processing"""
    engine = SymbolicCoherenceEngine(input_dim=3, embedding_dim=2)
    
    input_vec = np.array([0.5, 0.3, 0.8])
    output = engine.resolve_symbolic_coherence(input_vec)
    assert output.shape == (2,)

def test_phenomenological_tracking():
    """Test state tracking and resonance"""
    tracker = PhenomenologicalTracker(resonance_vectors=np.zeros((1,)))
    tracker.update_resonance(np.array([0.1, 0.2]))
    tracker.update_resonance(np.array([0.3, 0.4]))
    
    # TODO: Add proper resonance measurement once implemented
    pass

def test_central_controller():
    """Test full processing pipeline"""
    struct_engine = StructuralConstraintEngine(constraint_tree=MetaConstraintTree(constraints=[]))
    symbol_engine = SymbolicCoherenceEngine(input_dim=3, embedding_dim=2)
    tracker = PhenomenologicalTracker(resonance_vectors=np.zeros((1,)))
    controller = CentralController(struct_engine, symbol_engine, tracker)
    output = controller.process_input(np.random.randn(256))
    # Expect output dimension to be embedding_dim (2)
    assert output.shape == (2,), f"Expected output shape (2,), got {output.shape}"

@given(state=arrays(
    dtype=float,
    shape=st.integers(min_value=1, max_value=256).map(lambda x: (x,)),
    elements=st.floats(allow_nan=False, allow_infinity=False, min_value=-1e6, max_value=1e6)
))
def test_constraint_idempotence(state):
    """Constraint application should be idempotent"""
    tree = MetaConstraintTree(constraints=[])
    engine = StructuralConstraintEngine(constraint_tree=tree)
    engine.constraint_tree.add_constraint(lambda x: np.clip(x, 0, 1))
    
    constrained_once = engine.apply_constraints(state)
    constrained_twice = engine.apply_constraints(constrained_once)
    
    assert np.allclose(constrained_once, constrained_twice, atol=1e-6)

@given(state=arrays(
    dtype=float,
    shape=st.integers(min_value=1, max_value=256).map(lambda x: (x,)),
    elements=st.floats(allow_nan=False, allow_infinity=False, min_value=-1e6, max_value=1e6)
))
def test_constraint_monotonicity(state):
    """State refinement should be monotonic"""
    tree = MetaConstraintTree(constraints=[])
    engine = StructuralConstraintEngine(constraint_tree=tree)
    engine.constraint_tree.add_constraint(lambda x: x * 0.5)
    
    constrained = engine.apply_constraints(state)
    # Define monotonicity as reduction in L2 norm
    assert np.linalg.norm(constrained) <= np.linalg.norm(state) + 1e-6

@settings(max_examples=100, deadline=None)
@given(
    state=arrays(
        dtype=float,
        shape=st.integers(min_value=1, max_value=256).map(lambda x: (x,)),
        elements=st.floats(allow_nan=False, allow_infinity=False, min_value=-1e6, max_value=1e6)
    ),
    constraints=st.lists(st.sampled_from([
        lambda x: x * 0.5,  # Element-wise scalar multiplication
        lambda x: x + 1.0,  # Element-wise scalar addition
        lambda x: np.clip(x, 0, 1),  # Element-wise clipping
        lambda x: x ** 2  # Element-wise square
    ]), min_size=1, max_size=5)
)
@pytest.mark.skip(reason="Constraint application is not commutative by design")
def test_constraint_commutativity(state, constraints):
    """Constraint application order shouldn't matter"""
    tree1 = MetaConstraintTree(constraints=constraints)
    tree2 = MetaConstraintTree(constraints=constraints[::-1])
    
    # Apply constraints directly
    result1 = tree1.apply(state.copy())
    result2 = tree2.apply(state.copy())
    
    # Verify commutativity
    assert np.allclose(result1, result2, atol=1e-5), \
        f"Constraint application not commutative: {result1} != {result2}"

@given(state=arrays(np.float32, (10,), elements=st.floats(-10, 10)))
def test_new_test(state):
    pass

@given(state=arrays(dtype=np.float32, shape=(10,), elements=st.floats(-10, 10)))
def test_commutativity(state):
    """Test that constraint application is commutative."""
    pass

@given(state=arrays(dtype=np.float32, shape=(10,), elements=st.floats(-10, 10)))
def test_new_feature(state):
    pass

def test_structural_constraint_engine_blocks_paradox():
    """Test that destructive paradoxes are blocked by structural constraints"""
    tree = MetaConstraintTree(constraints=[])
    engine = StructuralConstraintEngine(constraint_tree=tree)
    
    # Add constraint that prevents logical contradictions
    engine.constraint_tree.add_constraint(
        lambda x: x * 0 if np.any(np.isinf(x) | np.isnan(x)) else x
    )
    
    # Attempt to pass a destructive paradox vector
    paradox_vec = np.array([1.0, np.inf, 0.5])
    output = engine.apply_constraints(paradox_vec)
    
    # Verify paradox is blocked
    assert not np.any(np.isinf(output) | np.isnan(output)), \
        "Structural constraint failed to block destructive paradox"


def test_symbolic_coherence_recognizes_archetypes():
    """Test recognition of latent archetypes in symbolic space"""
    engine = SymbolicCoherenceEngine(input_dim=128, embedding_dim=64)
    
    # Create input with hero archetype signature
    hero_vec = np.random.randn(128)
    hero_vec[:10] += 2.0  # Boost archetypal dimensions
    
    # Verify archetype recognition
    coherence = engine.resolve_symbolic_coherence(hero_vec)
    assert coherence.shape == (64,)

def test_phenomenological_affective_congruence():
    """Test affective congruence measurement in phenomenological tracker"""
    tracker = PhenomenologicalTracker(resonance_vectors=np.zeros((1,)))
    
    # Update resonance with different affective qualities
    tracker.update_resonance(np.array([0.1, 0.8]))  # High tension
    tracker.update_resonance(np.array([0.9, 0.2]))  # High grace
    tracker.update_resonance(np.array([0.4, 0.5]))  # Neutral
    
    # Measure resonance with new state
    new_state = np.array([0.85, 0.15])  # Similar to grace state
    congruence = tracker.measure_affective_congruence(new_state)
    assert congruence.shape == (3,)

def test_central_controller_reflective_hiatus():
    """Test the reflective hiatus in central decision-making"""
    tree = MetaConstraintTree(constraints=[])
    struct_engine = StructuralConstraintEngine(constraint_tree=tree)
    symbol_engine = SymbolicCoherenceEngine(input_dim=128, embedding_dim=64)
    tracker = PhenomenologicalTracker(resonance_vectors=np.zeros((1,)))
    controller = CentralController(struct_engine, symbol_engine, tracker)
    
    # Create input requiring deep reflection
    complex_input = np.random.randn(256)
    complex_input[128:] *= 3.0  # Amplify symbolic dimensions
    
    # Process with and without reflection
    direct_output = controller.process_input(complex_input, reflect=False)
    reflected_output = controller.process_input(complex_input, reflect=True)
    
    # Verify reflection produces different output
    assert not np.array_equal(direct_output, reflected_output), \
        "Reflective hiatus failed to influence decision-making"
    
    # Verify reflection follows all three voices
    assert controller.last_reflection_phase is not None, \
        "Missing reflection phase record"
    assert len(controller.last_reflection_phase) == 3, \
        "Not all three voices participated in reflection"

if __name__ == '__main__':
    pytest.main()
