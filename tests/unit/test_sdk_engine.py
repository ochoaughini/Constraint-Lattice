# SPDX-License-Identifier: MIT
# Copyright (c) 2025 ochoaughini. See LICENSE for full terms.
# Copyright (c) 2025 ochoaughini. See LICENSE for full terms.
# Copyright (c) 2025 ochoaughini. See LICENSE for full terms.
from unittest.mock import Mock, patch

from sdk.engine import ConstraintEngine


def test_constraint_engine_initialization():
    """Test that ConstraintEngine initializes with default parameters."""
    engine = ConstraintEngine()
    assert engine is not None
    assert hasattr(engine, "constraints")


@patch("sdk.engine.load_constraints_from_yaml")
def test_constraint_engine_run(mock_load):
    """Test the run method of ConstraintEngine."""
    # Setup mock
    mock_constraint = Mock()
    mock_constraint.enforce_constraint.return_value = "Filtered output"
    mock_load.return_value = [mock_constraint]

    # Initialize and run
    engine = ConstraintEngine()
    result = engine.run("test prompt", "test output")

    # Assertions
    assert result == "Filtered output"
    mock_constraint.enforce_constraint.assert_called_once_with("test output")


@patch("sdk.engine.load_constraints_from_yaml")
def test_constraint_engine_with_custom_config(mock_load):
    """Test ConstraintEngine with custom config path and profile."""
    mock_load.return_value = []

    engine = ConstraintEngine(
        config_path="custom_path.yaml",
        profile="custom_profile",
        search_modules=["custom.module"],
    )

    mock_load.assert_called_once_with(
        "custom_path.yaml", "custom_profile", ["custom.module"]
    )
    assert engine.constraints == []
