# SPDX-License-Identifier: MIT
# Copyright (c) 2025 ochoaughini. See LICENSE for full terms.
# Copyright (c) 2025 ochoaughini. See LICENSE for full terms.
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class IntrospectiveLayer:
    def __init__(self, memory_stream):
        self.memory_stream = memory_stream
        self.reflection_threshold = 0.4
    
    def generate_reflection(self, current_vector, current_narrative):
        """Generate reflective commentary on current state"""
        # Compare with historical states
        similar_memories = self.memory_stream.recall(current_vector, top_k=3)
        
        if not similar_memories:
            return "This is a novel experience with no historical parallels"
        
        # Analyze similarity patterns
        similarities = [cosine_similarity([current_vector], [np.array(mem['vector'])])[0][0] 
                        for mem in similar_memories]
        avg_similarity = np.mean(similarities)
        
        if avg_similarity > self.reflection_threshold:
            return (f"This echoes {similar_memories[0]['narrative']} from the past, "
                    f"suggesting a recurring pattern of {current_narrative}")
        else:
            return (f"This diverges from historical patterns like {similar_memories[0]['narrative']}, "
                    f"marking a departure towards {current_narrative}")
