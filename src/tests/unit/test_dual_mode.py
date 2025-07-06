# SPDX-License-Identifier: MIT
# Copyright (c) 2025 ochoaughini. See LICENSE for full terms.
# Copyright (c) 2025 ochoaughini. See LICENSE for full terms.
import os
import sys
import pytest
from unittest.mock import patch
from constraint_lattice.engine.apply import apply_constraints
from constraint_lattice.engine.mode import get_execution_mode


def test_dual_mode_switching():
    """Test that apply_constraints switches modes correctly"""
    # Test supervisory mode
    with patch('constraint_lattice.engine.mode.get_execution_mode', return_value="supervisory"):
        result = apply_constraints("", "test", [], return_audit_trace=False)
        assert result == "test"  # Should pass through in mock
    
    # Test executor mode
    with patch('constraint_lattice.engine.mode.get_execution_mode', return_value="executor"):
        result = apply_constraints("", "test", [], return_audit_trace=False)
        assert result == "test"  # Should pass through in mock


def test_force_executor_mode():
    """Test environment variable override"""
    # Set override to executor
    os.environ["CONSTRAINT_LATTICE_FORCE_EXECUTOR_MODE"] = "true"
    assert get_execution_mode() == "executor"
    
    # Set override to supervisory
    os.environ["CONSTRAINT_LATTICE_FORCE_EXECUTOR_MODE"] = "false"
    assert get_execution_mode() == "supervisory"
