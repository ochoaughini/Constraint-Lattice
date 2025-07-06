"""
Base Constraint class

This module defines the abstract base class for all constraints.
"""
from abc import ABC, abstractmethod


class Constraint(ABC):
    """
    Abstract base class for constraints.

    All constraints must inherit from this class and implement the `process_text` method.
    """
    priority: int = 50

    @abstractmethod
    def process_text(self, text: str) -> str:
        """
        Process the input text according to the constraint.

        Args:
            text: The input text to process

        Returns:
            The processed text
        """
        pass
