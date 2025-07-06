# SPDX-License-Identifier: MIT
# Copyright (c) 2025 ochoaughini. See LICENSE for full terms.
# Copyright (c) 2025 ochoaughini. See LICENSE for full terms.
from unittest.mock import Mock, patch

from sdk.engine import ConstraintEngine


def test_constraint_engine_initialization():
    """Test that ConstraintEngine initializes with default parameters."""
    engine = ConstraintEngine()
    assert engine is not None
    assert hasattr(engine, "constraints")


def test_constraint_engine_run():
    """Test the run method of ConstraintEngine."""
    # Setup mock
    mock_constraint = Mock()
    mock_constraint.process_text.return_value = "Filtered output"
    
    # Initialize and run
    engine = ConstraintEngine(constraints=[mock_constraint])
    result = engine.run("test prompt", "test output")
    
    # Assertions
    assert result == "Filtered output"
    mock_constraint.process_text.assert_called_once_with("test output")


def test_constraint_engine_with_custom_config():
    """Test ConstraintEngine with custom config path and profile."""
    # Setup mock
    mock_constraint = Mock()
    
    # Initialize with custom config
    engine = ConstraintEngine(
        config_path="custom_path.yaml",
        profile="custom_profile",
        search_modules=["custom.module"],
        constraints=[mock_constraint]
    )
    
    # Assertions
    assert engine.config_path == "custom_path.yaml"
    assert engine.profile == "custom_profile"
    assert engine.search_modules == ["custom.module"]
