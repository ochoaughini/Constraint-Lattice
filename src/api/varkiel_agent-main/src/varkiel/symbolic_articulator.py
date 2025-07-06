# SPDX-License-Identifier: MIT
# Copyright (c) 2025 ochoaughini. See LICENSE for full terms.
# Copyright (c) 2025 Lexsight LCC. All rights reserved.
# See saas/LICENSE-BSL.txt for full terms.
# Copyright (c) 2025 Lexsight LCC. All rights reserved.
# See saas/LICENSE-BSL.txt for full terms.
import numpy as np

class SymbolicArticulator:
    def __init__(self, archetype_map):
        self.archetype_map = {}
        for archetype, vector in archetype_map.items():
            # Ensure vector is 128-dimensional
            if len(vector) != 128:
                # If not, pad or truncate
                if len(vector) < 128:
                    padded = np.pad(vector, (0, 128 - len(vector)), 'constant')
                    self.archetype_map[archetype] = padded
                else:
                    self.archetype_map[archetype] = vector[:128]
            else:
                self.archetype_map[archetype] = vector
        
        self.metaphor_templates = [
            "The {primary} resonates with echoes of {secondary}",
            "{primary} emerges from the shadow of {secondary}",
            "A dance of {primary} and {secondary} unfolds",
            "{primary} confronts the absence of {secondary}",
            "In the space between {primary} and {secondary}, meaning forms"
        ]
    
    def vector_to_narrative(self, vector: np.ndarray) -> str:
        """Transform vector into archetypal narrative"""
        # Find closest archetypes
        archetypes = self._find_closest_archetypes(vector, top_k=2)
        
        # Select metaphor template based on vector characteristics
        vector_norm = np.linalg.norm(vector)
        template_idx = int(vector_norm * len(self.metaphor_templates)) % len(self.metaphor_templates)
        
        return self.metaphor_templates[template_idx].format(
            primary=archetypes[0][0],
            secondary=archetypes[1][0]
        )
    
    def _find_closest_archetypes(self, vector, top_k=3):
        """Find top-k closest archetypes to vector"""
        similarities = {}
        for archetype, base_vector in self.archetype_map.items():
            sim = np.dot(vector, base_vector) / (np.linalg.norm(vector) * np.linalg.norm(base_vector))
            similarities[archetype] = sim
        return sorted(similarities.items(), key=lambda x: x[1], reverse=True)[:top_k]
