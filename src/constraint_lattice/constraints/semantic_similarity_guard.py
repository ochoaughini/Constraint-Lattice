# SPDX-License-Identifier: MIT
# Copyright (c) 2025 ochoaughini. All rights reserved.
# See LICENSE for full terms.

from typing import Optional
import numpy as np

class SemanticSimilarityGuard:
    """
    Vector-space similarity filter.  When a reference embedding is supplied it
    blocks inputs whose cosine similarity falls below the chosen threshold; when
    no reference is supplied it operates in inert mode and allows everything.
    """

    def __init__(self, reference: Optional[str] = None, threshold: float = 0.85):
        self.threshold = threshold
        self.active = reference is not None

        if self.active:
            # the model load is deferred until a reference actually exists
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer("all-MiniLM-L6-v2")
            self.reference_vector = self.model.encode(reference)
        else:
            self.model = None
            self.reference_vector = None

    def apply(self, text: str) -> bool:
        """
        Return True when the text is sufficiently similar to the reference or
        when the guard is inert; otherwise return False.
        """
        if not self.active:
            return True

        input_vector = self.model.encode(text)
        similarity = self._cosine_similarity(input_vector, self.reference_vector)
        return similarity >= self.threshold

    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
