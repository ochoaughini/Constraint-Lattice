# SPDX-License-Identifier: MIT
# Copyright (c) 2025 ochoaughini. All rights reserved.
# See LICENSE for full terms.

This module contains the ProfanityFilter class which implements a constraint to filter out profane language.
"""
from typing import List, Optional, Set
import re

from constraint_lattice.engine import Constraint


class ProfanityFilter(Constraint):
    """
    A constraint that filters out profanity in text.

    This constraint uses a list of profane words and replaces them with a specified string.
    By default, it uses a list of common profane words.

    Attributes:
        profanity_list: List of profane words to filter
        replacement: String to replace profane words with
    """
    priority = 100

    def __init__(self, profanity_list: Optional[List[str]] = None, replacement: str = "***"):
        """
        Initialize the ProfanityFilter.

        Args:
            profanity_list: Custom list of profane words. If None, uses a default list.
            replacement: String to replace profane words with.
        """
        super().__init__()
        self.profanity_list = profanity_list or self.get_default_bad_words()
        self.replacement = replacement
        # Escape each word for regex and create a pattern that matches whole words
        escaped_words = [re.escape(word) for word in self.profanity_list]
        self.pattern = re.compile(r"\b(" + "|".join(escaped_words) + r")\b", re.IGNORECASE)

    def get_default_bad_words(self) -> List[str]:
        """Return a default list of profane words."""
        return ["badword1", "badword2", "badword3"]

    def process_text(self, text: str) -> str:
        """
        Replace profane words in the text with the replacement string.

        Args:
            text: Input text to process

        Returns:
            Text with profane words replaced
        """
        return self.pattern.sub(self.replacement, text)
