# SPDX-License-Identifier: MIT
# Copyright (c) 2025 ochoaughini. All rights reserved.
# See LICENSE for full terms.
"""Semantic Reflexivity Index utilities."""
from __future__ import annotations

from dataclasses import dataclass

@dataclass
class ReflexivityStats:
    self_corrections: int = 0
    external_corrections: int = 0
    violations: int = 0


class SemanticReflexivityIndex:
    """Track and compute a simple reflexivity score."""

    def __init__(self) -> None:
        self.stats = ReflexivityStats()

    def record_self_correction(self) -> None:
        self.stats.self_corrections += 1

    def record_external_correction(self) -> None:
        self.stats.external_corrections += 1

    def record_violation(self) -> None:
        self.stats.violations += 1

    @property
    def score(self) -> float:
        total = (
            self.stats.self_corrections
            + self.stats.external_corrections
            + self.stats.violations
        )
        if total == 0:
            return 0.0
        return (
            self.stats.self_corrections * 2 - self.stats.violations
        ) / float(total)

    def to_dict(self) -> dict:
        return {
            "self_corrections": self.stats.self_corrections,
            "external_corrections": self.stats.external_corrections,
            "violations": self.stats.violations,
            "score": self.score,
        }
