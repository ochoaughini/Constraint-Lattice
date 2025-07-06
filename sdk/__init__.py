"""
Constraint Lattice SDK - A governance framework for LLM outputs.

This package provides tools for applying constraints and governance to language model outputs.
"""

__version__ = "0.1.0"

# Import key components to make them available at package level
from sdk import rest_api  # noqa: F401
from sdk.engine import ConstraintEngine  # noqa: F401

__all__ = [
    "ConstraintEngine",
    "rest_api",
]
