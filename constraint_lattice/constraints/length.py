# SPDX-License-Identifier: MIT
# Copyright (c) 2025 ochoaughini. All rights reserved.
# See LICENSE for full terms.

@constraint(priority=50)
class LengthConstraint:
    """Constraint for enforcing maximum text length.

    Attributes:
        max_length: Maximum allowed character length
        truncate: Whether to truncate text that exceeds max_length
        ellipsis: String to append when truncating
    """

    def __init__(self, max_length: int, truncate: bool = True, ellipsis: str = "[...]") -> None:
        """Initialize length constraint.

        Args:
            max_length: Maximum allowed character length
            truncate: Whether to truncate text that exceeds max_length
            ellipsis: String to append when truncating
        """
        if max_length <= 0:
            raise ValueError("max_length must be greater than 0")

        self.max_length = max_length
        self.truncate = truncate
        self.ellipsis = ellipsis
        logging.debug(f"Initialized {self}")

    def process_text(self, text: str) -> str:
        """Apply length constraint to input text.

        Args:
            text: Input text to process

        Returns:
            Text that conforms to length constraint
        """
        if len(text) <= self.max_length:
            return text

        if not self.truncate:
            raise ValueError(
                f"Text exceeds maximum length of {self.max_length} characters"
            )

        # Calculate available space for text after accounting for ellipsis
        available_length = self.max_length - len(self.ellipsis)
        if available_length <= 0:
            return self.ellipsis[:self.max_length]

        return text[:available_length] + self.ellipsis

    def __call__(self, text: str) -> str:
        """Alias for process_text method.

        Args:
            text: Input text to process

        Returns:
            Text that conforms to length constraint
        """
        return self.process_text(text)

    def __repr__(self) -> str:
        """Return string representation of the constraint.

        Returns:
            String representation
        """
        return (
            f"LengthConstraint(max_length={self.max_length}, "
            f"truncate={self.truncate}, ellipsis='{self.ellipsis}')"
        )