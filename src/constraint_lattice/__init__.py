# SPDX-License-Identifier: MIT
# Copyright (c) 2025 ochoaughini. See LICENSE for full terms.
# Copyright (c) 2025 ochoaughini. See LICENSE for full terms.
"""
Constraint Lattice - Symbolic Reasoning Engine

Top-level package for constraint-based AI systems.
"""

__all__ = [
    # Core engine components
    'apply_constraints',
    'ConstraintConfig',
    
    # Important classes
    'AuditStep',
    'AuditTrace',
    
    # Submodules
    'engine',
    'constraints',
    'services'
]

# Public API imports
from .engine import apply_constraints
from .engine.schema import ConstraintConfig
from .engine.apply import AuditStep, AuditTrace

from __future__ import annotations

import importlib
import sys
from types import ModuleType

def _alias(top_level_name: str) -> None:
    """Import *top_level_name* and alias it under constraint_lattice.*."""
    try:
        module = importlib.import_module(top_level_name)
    except (ImportError, ModuleNotFoundError):  # optional or gated module
        return
    sys.modules[f"constraint_lattice.{top_level_name}"] = module
    setattr(sys.modules[__name__], top_level_name, module)

for _m in ("engine", "constraints", "sdk", "saas"):
    _alias(_m)

# Expose a PEP 561 marker so type-checkers know this is a typed pkg in future.
try:
    from importlib.metadata import version

    __version__ = version("constraint-lattice")
except Exception:  # pragma: no cover
    __version__ = "0.0.0-dev"
