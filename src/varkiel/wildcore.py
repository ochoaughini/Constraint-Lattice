# SPDX-License-Identifier: MIT
"""Minimal WildCore anomaly detector."""

from __future__ import annotations

from typing import Iterable


class WildCore:
    """Detect banned phrases in text."""

    def __init__(self, banned: Iterable[str] | None = None) -> None:
        self._banned = set(banned or [])

    def scan(self, text: str) -> bool:
        """Return True if *text* contains banned content."""
        lowered = text.lower()
        return any(word in lowered for word in self._banned)
