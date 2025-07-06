# SPDX-License-Identifier: MIT
# Copyright (c) 2025 ochoaughini. All rights reserved.
# See LICENSE for full terms.

"""This module contains the ``LengthConstraint`` class which enforces a
maximum length on text."""

from constraint_lattice.engine import Constraint


class LengthConstraint(Constraint):
    """
    A constraint that enforces a maximum length on text.

    Attributes:
        max_length: The maximum allowed length for the text
        truncate: Whether to truncate the text if it exceeds the max_length
        ellipsis: The string to append to truncated text
    """

    priority = 50

    def __init__(self, max_length: int, truncate: bool = True, ellipsis: str = ""):
        """
        Initialize the LengthConstraint.

        Args:
            max_length: Maximum allowed length for the text
            truncate: Whether to truncate the text if it exceeds the max_length
            ellipsis: The string to append to truncated text
        """
        super().__init__()
        self.max_length = max_length
        self.truncate = truncate
        self.ellipsis = ellipsis

    def process_text(self, text: str) -> str:
        """
        Truncate text to the maximum length if it exceeds the limit.

        Args:
            text: The input text to process

        Returns:
            The truncated text if it was too long, otherwise the original text
        """
        if len(text) > self.max_length and self.truncate:
            truncated = text[: self.max_length]
            if self.ellipsis:
                truncated += self.ellipsis
            return truncated
        return text
