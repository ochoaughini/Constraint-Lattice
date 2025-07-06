# SPDX-License-Identifier: MIT
# Copyright (c) 2025 ochoaughini. See LICENSE for full terms.
# Copyright (c) 2025 ochoaughini. See LICENSE for full terms.
import numpy as np

class SymbolicInterpreter:
    def __init__(self, symbol_map):
        self.symbol_map = symbol_map  # {symbol: vector}
        
    def vector_to_symbols(self, vector: np.ndarray, top_k=3) -> list:
        """Convert vector to top-k symbolic concepts"""
        similarities = {}
        for symbol, base_vector in self.symbol_map.items():
            sim = np.dot(vector, base_vector) / (np.linalg.norm(vector) * np.linalg.norm(base_vector))
            similarities[symbol] = sim
        return sorted(similarities.items(), key=lambda x: x[1], reverse=True)[:top_k]
