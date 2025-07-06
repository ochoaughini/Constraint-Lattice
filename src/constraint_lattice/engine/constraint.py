# SPDX-License-Identifier: MIT
"""Abstract base class for constraints."""

from abc import ABC, abstractmethod


class Constraint(ABC):
    """Base class that all constraints must inherit."""

    priority: int = 50

    @abstractmethod
    def process_text(self, text: str) -> str:
        """Process the input text according to the constraint."""
        pass
