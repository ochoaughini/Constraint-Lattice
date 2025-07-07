# SPDX-License-Identifier: MIT
# Copyright (c) 2025 ochoaughini. All rights reserved.
# See LICENSE for full terms.

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

    # Advanced modules
    'ConstraintOntologyCompiler',
    'CrossAgentAlignmentLedger',
    'SemanticReflexivityIndex',

    # Submodules
    'engine',
    'constraints',
    'services',
    'compiler',
    'ledger',
    'reflexivity'
]

# Public API imports
from .engine import apply_constraints
from .engine.schema import ConstraintConfig
from .engine.apply import AuditStep, AuditTrace
from .compiler import ConstraintOntologyCompiler
from .ledger import CrossAgentAlignmentLedger
from .reflexivity import SemanticReflexivityIndex

import importlib
import sys
import os
from types import ModuleType

def _alias(top_level_name: str) -> None:
    """Import *top_level_name* and alias it under constraint_lattice.*."""
    try:
        module = importlib.import_module(top_level_name)
    except (ImportError, ModuleNotFoundError):  # optional or gated module
        return
    sys.modules[f"constraint_lattice.{top_level_name}"] = module
    setattr(sys.modules[__name__], top_level_name, module)

for _m in ("engine", "constraints"):
    _alias(_m)
if os.getenv("CL_IMPORT_SDK", "1") == "1":
    _alias("sdk")

if os.getenv("ENABLE_SAAS_FEATURES", "false").lower() in ["true", "1"]:
    from .saas import *

# Expose a PEP 561 marker so type-checkers know this is a typed pkg in future.
try:
    from importlib.metadata import version

    __version__ = version("constraint-lattice")
except Exception:  # pragma: no cover
    __version__ = "0.0.0-dev"
