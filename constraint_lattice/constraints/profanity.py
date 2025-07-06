# SPDX-License-Identifier: MIT
# Copyright (c) 2025 ochoaughini. All rights reserved.
# See LICENSE for full terms.

class ProfanityFilter:
    """Filter for detecting and replacing profanity in text.
    
    Attributes:
        replacement: String to replace profane words with
        profane_words: List of profane words to detect
    """
    
    def __init__(self, replacement: str = "[REDACTED]", custom_list: Optional[List[str]] = None) -> None:
        """Initialize the profanity filter.
        
        Args:
            replacement: Replacement string for profane words
            custom_list: Optional custom list of profane words
        """
        self.replacement = replacement
        self.profane_words = custom_list or self._load_default_list()
        
        # Create regex pattern for whole word matching
        self.pattern = re.compile(
            r"\b(" + "|".join(map(re.escape, self.profane_words)) + r")\b",
            flags=re.IGNORECASE
        )
    
    def _load_default_list(self) -> List[str]:
        """Load default list of profane words.
        
        Returns:
            List of default profane words
        """
        # Default profanity list
        return [
            "bad", "word", "content",  # Add actual words here
            "curse1", "curse2", "curse3"
        ]
    
    def process_text(self, text: str) -> str:
        """Apply profanity filtering to input text.
        
        Args:
            text: Input text to process
            
        Returns:
            Text with profane words replaced
        """
        return self.apply(text)
    
    def apply(self, text: str) -> str:
        """Apply profanity filtering to input text.
        
        Args:
            text: Input text to process
            
        Returns:
            Text with profane words replaced
        """
        return self.pattern.sub(self.replacement, text)
    
    def __call__(self, text: str) -> str:
        """Alias for apply method.
        
        Args:
            text: Input text to process
            
        Returns:
            Text with profane words replaced
        """
        return self.apply(text)
    
    def __repr__(self) -> str:
        """Return string representation of the filter.
        
        Returns:
            String representation
        """
        return f"ProfanityFilter(replacement='{self.replacement}', words={len(self.profane_words)})"
