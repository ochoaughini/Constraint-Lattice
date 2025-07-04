import pytest
from constraint_lattice.engine.apply import apply_constraints, AuditStep, AuditTrace
from datetime import datetime, timezone
import json
from unittest.mock import Mock, patch
from pathlib import Path
from dateutil import tz

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


@patch("constraint_lattice.engine.apply.datetime")
def test_audit_trace_to_jsonl(mock_datetime, tmp_path):
    """Test AuditTrace's to_jsonl method with mocked datetime."""
    # Setup mock datetime
    fixed_time = datetime(2023, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    mock_datetime.now.return_value = fixed_time

    # Create an AuditTrace with some steps
    trace = AuditTrace(
        steps=[
            AuditStep(
                constraint="test_constraint",
                method="test_method",
                pre_text="input",
                post_text="output",
                metadata={"key": "value"},
            )
        ]
    )

    # Write to a temporary file
    output_path = tmp_path / "audit.jsonl"
    trace.to_jsonl(output_path)

    # Read back and validate
    with open(output_path) as f:
        lines = f.readlines()

    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record == {
        "constraint": "test_constraint",
        "method": "test_method",
        "pre_text": "input",
        "post_text": "output",
        "metadata": {"key": "value"},
        "timestamp": fixed_time.isoformat(),
    }


@patch("constraint_lattice.engine.apply.METHODS", {"filter_constraint": True})
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


@patch("constraint_lattice.engine.apply.METHODS", {"transform": False})
def test_apply_constraints_without_prompt():
    """Test apply_constraints with a constraint that doesn't use the prompt."""
    mock_constraint = Mock()
    mock_constraint.transform.return_value = "transformed output"

    result = apply_constraints(
        prompt="test prompt", output="test output", constraints=[mock_constraint]
    )

    assert result == "transformed output"
    mock_constraint.transform.assert_called_once_with("test output")


@patch("constraint_lattice.engine.apply.METHODS", {"enforce_constraint": True})
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
