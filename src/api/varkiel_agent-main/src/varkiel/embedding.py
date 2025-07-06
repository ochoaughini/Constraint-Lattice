# SPDX-License-Identifier: MIT
# Copyright (c) 2025 ochoaughini. See LICENSE for full terms.
# Copyright (c) 2025 Lexsight LCC. All rights reserved.
# See saas/LICENSE-BSL.txt for full terms.
# Copyright (c) 2025 Lexsight LCC. All rights reserved.
# See saas/LICENSE-BSL.txt for full terms.
"""
Varkiel Agent - Advanced AI Constraint System
Copyright (C) 2025 Lexsight LLC
SPDX-License-Identifier: AGPL-3.0-only OR Commercial
"""

from sentence_transformers import SentenceTransformer
import torch
import numpy as np
import time

class SymbolicEmbedder:
    """Dense semantic embedding using transformer models"""
    def __init__(self, model_name: str = 'all-mpnet-base-v2'):
        self.model = SentenceTransformer(model_name)
        self.last_processing_time = 0.0
        
    def embed(self, text: str) -> np.ndarray:
        """Generate embedding for input text"""
        start_time = time.time()
        embedding = self.model.encode(text)
        self.last_processing_time = time.time() - start_time
        return embedding

    def batch_embed(self, texts: list) -> np.ndarray:
        """Generate embeddings for a batch of texts"""
        return np.vstack([self.embed(text) for text in texts])

    def __call__(self, text: str) -> np.ndarray:
        """Alias for embed()"""
        return self.embed(text)
