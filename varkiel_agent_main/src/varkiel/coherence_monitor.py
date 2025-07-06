# SPDX-License-Identifier: MIT
# Copyright (c) 2025 ochoaughini. See LICENSE for full terms.
# Copyright (c) 2025 ochoaughini. See LICENSE for full terms.
import numpy as np

class RecursiveInvarianceMonitor:
    def __init__(self):
        self.history = []
        
    def evaluate(self, current_output) -> float:
        """Calculate coherence between current and previous outputs"""
        if len(self.history) > 0:
            prev = self.history[-1]
            coherence = np.dot(current_output, prev) / (np.linalg.norm(current_output) * np.linalg.norm(prev))
            self.history.append(current_output)
            return coherence
        self.history.append(current_output)
        return 1.0  # Perfect coherence for first output
