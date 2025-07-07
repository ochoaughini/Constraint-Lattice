# SPDX-License-Identifier: MIT
"""Simple drift detection stub."""

from __future__ import annotations

from typing import Iterable


class DriftManager:
    """Track distribution of added knowledge."""

    def __init__(self) -> None:
        self.count = 0

    def update(self, items: Iterable[str]) -> None:
        self.count += len(list(items))

    def drift_score(self) -> float:
        return float(self.count)
