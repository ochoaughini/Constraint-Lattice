# SPDX-License-Identifier: MIT
"""Length constraint utilities used in unit tests."""
from __future__ import annotations

from dataclasses import dataclass
import logging

from src.constraint_lattice.engine.scheduler import constraint

logger = logging.getLogger(__name__)


@constraint(priority=50)
@dataclass
class LengthConstraint:
    """Enforce a maximum length on text."""

    max_length: int
    truncate: bool = True
    ellipsis: str = "[...]"

    def __post_init__(self) -> None:
        if self.max_length <= 0:
            raise ValueError("max_length must be greater than 0")
        logger.debug("initialised %s", self)

    def process_text(self, text: str) -> str:
        if len(text) <= self.max_length:
            return text
        if not self.truncate:
            raise ValueError(
                f"Text exceeds maximum length of {self.max_length} characters"
            )
        available = self.max_length - len(self.ellipsis)
        if available <= 0:
            return self.ellipsis[: self.max_length]
        return text[:available] + self.ellipsis

    def __call__(self, text: str) -> str:  # pragma: no cover - convenience wrapper
        return self.process_text(text)

    def __repr__(self) -> str:
        return (
            f"LengthConstraint(max_length={self.max_length}, "
            f"truncate={self.truncate}, ellipsis='{self.ellipsis}')"
        )
