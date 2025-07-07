# SPDX-License-Identifier: MIT
# Copyright (c) 2025 ochoaughini. All rights reserved.
# See LICENSE for full terms.
from __future__ import annotations

"""Constraint Ontology Compiler module."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
import yaml

# Supported constraint types - kept intentionally small for demonstration
ALLOWED_TYPES = {"regex", "text", "semantic", "style", "safety"}

@dataclass
class CompiledConstraint:
    """Normalized representation of a constraint entry."""

    name: str
    type: str
    params: Dict[str, str] = field(default_factory=dict)
    category: Optional[str] = None
    severity: Optional[str] = None
    contexts: List[str] = field(default_factory=list)


def _validate_entry(entry: Dict[str, object]) -> None:
    if entry.get("type") not in ALLOWED_TYPES:
        raise ValueError(f"Unknown constraint type: {entry.get('type')}")
    if "name" not in entry:
        raise ValueError("Constraint entry missing 'name'")


class ConstraintOntologyCompiler:
    """Compile human-friendly constraint definitions into structured objects."""

    def compile_file(self, path: str) -> List[CompiledConstraint]:
        """Load and compile a YAML constraints file."""
        with open(path, "r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
        entries = data.get("constraints", [])
        compiled: List[CompiledConstraint] = []
        for raw in entries:
            _validate_entry(raw)
            compiled.append(
                CompiledConstraint(
                    name=raw["name"],
                    type=raw["type"],
                    params=raw.get("params", {}),
                    category=raw.get("category"),
                    severity=raw.get("severity"),
                    contexts=list(raw.get("contexts", [])),
                )
            )
        return compiled
