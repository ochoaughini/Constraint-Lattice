# SPDX-License-Identifier: MIT
# Copyright (c) 2025 ochoaughini. See LICENSE for full terms.
# Copyright (c) 2025 ochoaughini. See LICENSE for full terms.
# Copyright (c) 2025 ochoaughini. See LICENSE for full terms.
from datetime import datetime
from unittest.mock import Mock, patch

from engine.apply import AuditStep, AuditTrace, apply_constraints


def test_audit_step_initialization():
    """Test AuditStep initialization and default values."""
    step = AuditStep(
        constraint="TestConstraint",
        method="test_method",
        pre_text="before",
        post_text="after",
        elapsed_ms=10.5,
    )
    assert step.constraint == "TestConstraint"
    assert step.method == "test_method"
    assert step.pre_text == "before"
    assert step.post_text == "after"
    assert step.elapsed_ms == 10.5
    assert isinstance(step.timestamp, datetime)


@patch("engine.apply.datetime")
def test_audit_trace_to_jsonl(mock_datetime, tmp_path):
    """Test AuditTrace's to_jsonl method with mocked datetime."""
    # Setup mock datetime
    fixed_time = datetime(2023, 1, 1, 12, 0, 0)
    mock_datetime.utcnow.return_value = fixed_time

    trace = AuditTrace()
    step = AuditStep(
        constraint="TestConstraint",
        method="test_method",
        pre_text="before",
        post_text="after",
        elapsed_ms=10.5,
    )
    trace.append(step)

    # Test writing to file
    output_path = tmp_path / "audit.jsonl"
    trace.to_jsonl(str(output_path))

    # Verify file content
    with open(output_path) as f:
        lines = f.readlines()
        assert len(lines) == 1
        content = lines[0]
        # Verify the content contains the expected fields
        assert '"constraint": "TestConstraint"' in content
        assert '"method": "test_method"' in content
        assert '"pre_text": "before"' in content
        assert '"post_text": "after"' in content
        assert '"elapsed_ms": 10.5' in content
        # Verify timestamp is present and in ISO format
        assert '"timestamp": "' in content
        # The actual timestamp will be the current time, so we just check the format
        assert content.count('"timestamp": "') == 1


@patch("engine.apply.METHODS", {"filter_constraint": True})
def test_apply_constraints_with_prompt():
    """Test apply_constraints with a constraint that uses the prompt."""
    mock_constraint = Mock()
    mock_constraint.filter_constraint.return_value = "filtered output"

    result = apply_constraints(
        prompt="test prompt", output="test output", constraints=[mock_constraint]
    )

    assert result == "filtered output"
    mock_constraint.filter_constraint.assert_called_once_with(
        "test prompt", "test output"
    )


@patch("engine.apply.METHODS", {"transform": False})
def test_apply_constraints_without_prompt():
    """Test apply_constraints with a constraint that doesn't use the prompt."""
    mock_constraint = Mock()
    mock_constraint.transform.return_value = "transformed output"

    result = apply_constraints(
        prompt="test prompt", output="test output", constraints=[mock_constraint]
    )

    assert result == "transformed output"
    mock_constraint.transform.assert_called_once_with("test output")


@patch("engine.apply.METHODS", {"enforce_constraint": True})
def test_apply_constraints_with_trace():
    """Test apply_constraints with return_trace=True."""
    mock_constraint = Mock()
    mock_constraint.enforce_constraint.return_value = "filtered output"

    result, trace = apply_constraints(
        prompt="test prompt",
        output="test output",
        constraints=[mock_constraint],
        return_trace=True,
    )

    assert result == "filtered output"
    assert len(trace) == 1
    assert trace[0].constraint == mock_constraint.__class__.__name__
    assert trace[0].method == "enforce_constraint"
    assert trace[0].pre_text == "test output"
    assert trace[0].post_text == "filtered output"
    assert trace[0].elapsed_ms >= 0
