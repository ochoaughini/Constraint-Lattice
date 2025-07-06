# SPDX-License-Identifier: MIT
# Copyright (c) 2025 ochoaughini. See LICENSE for full terms.
# Copyright (c) 2025 ochoaughini. See LICENSE for full terms.
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pytest
from unittest.mock import patch, MagicMock
from structural_constraint_engine import ConstraintType
from constraint_lattice_adapter import ConstraintLatticeAdapter, StateVector

class TestConstraintVectors:
    @pytest.fixture
    def adapter(self):
        # Mock lattice file
        return ConstraintLatticeAdapter("mock_lattice.json")
    
    def test_suspension_vector_no_paradox(self, adapter):
        """Test suspension vector when no paradox is detected"""
        state = np.array([0.5, 0.5, 0.5])
        result = adapter._apply_suspension(state)
        assert np.array_equal(result.state, state)
        
    def test_suspension_vector_with_paradox(self, adapter):
        """Test suspension vector when paradox is detected"""
        with patch.object(adapter, '_is_high_stakes_paradox', return_value=True):
            state = np.array([0.5, 0.5, 0.5])
            result = adapter._apply_suspension(state)
            assert np.array_equal(result.state, np.zeros_like(state))
            assert result.coherence_level == 0.0
        
    def test_overalignment_vector(self, adapter):
        """Test overalignment vector applies consensus factor"""
        adapter.security_consensus_factor = 0.8
        state = np.array([1.0, 1.0, 1.0])
        expected = state * 0.8
        result = adapter._apply_overalignment(state)
        assert np.array_equal(result.state, expected)
        
    def test_causal_erasure_vector(self, adapter):
        """Test causal erasure generalizes state"""
        with patch.object(adapter, '_generalize_state', return_value=np.array([0.5, 0.5, 0.5])):
            state = np.array([0.1, 0.5, 0.9])
            result = adapter._apply_causal_erasure(state)
            assert np.array_equal(result.state, np.array([0.5, 0.5, 0.5]))
        
    def test_apply_constraint_vector(self, adapter):
        """Test apply_constraint_vector delegates correctly"""
        with patch.object(adapter, '_apply_suspension') as suspension_spy:
            adapter.apply_constraint_vector(np.array([1,2,3]), ConstraintType.SUSPENSION)
            suspension_spy.assert_called_once()
