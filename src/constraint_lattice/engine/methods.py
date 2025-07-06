# SPDX-License-Identifier: MIT
# Copyright (c) 2025 ochoaughini. All rights reserved.
# See LICENSE for full terms.

These methods are used by the constraint engine.
"""
from typing import Callable, Dict, Optional, Tuple, Any


METHODS = {
    "process_text": False,  # Doesn't require prompt
    "process_with_prompt": True,  # Requires prompt
    
    # Legacy methods for backward compatibility
    "enforce_constraint": False,
    "filter_constraint": True,
    "regulate_constraint": True,
    "restrict_constraint": False,
    "suppress_constraint": False,
    "limit_constraint": False,
    "limit_questions_constraint": False,
    "enforce_tone_constraint": False,
    "monitor_constraint": False,
    "sanitize_constraint": False,
    "deny_constraint": False,
    "prevent_constraint": False,
    "redact_constraint": False,
    "nullify_constraint": False,
    "intervene_constraint": False,
}


def filter_constraint(
    prompt: str,
    output: str,
    constraint: Callable[[str], str]
) -> str:
    """Apply a filtering constraint to text output."""
    return constraint(output)


def transform(
    output: str,
    constraint: Callable[[str], str]
) -> str:
    """Apply a transformation constraint to text output."""
    return constraint(output)


def enforce_constraint(
    output: str,
    constraint: Callable[[str], bool]
) -> bool:
    """Enforce a boolean constraint on text output."""
    return constraint(output)


# Additional methods can be added below
