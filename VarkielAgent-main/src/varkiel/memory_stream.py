import numpy as np
import json
import os
from datetime import datetime
from collections import deque

class MemoryStream:
    def __init__(self, max_entries=1000, storage_path='memory.json'):
        self.storage_path = storage_path
        self.stream = deque(maxlen=max_entries)
        self.vector_index = {}
        self.load_memory()
    
    def record(self, input_text: str, output_vector: np.ndarray, narrative: str):
        """Record a new memory episode"""
        timestamp = datetime.now().isoformat()
        entry = {
            'timestamp': timestamp,
            'input': input_text,
            'vector': output_vector.tolist(),
            'narrative': narrative
        }
        self.stream.append(entry)
        self.vector_index[timestamp] = output_vector
        self.save_memory()
    
    def recall(self, query_vector: np.ndarray, top_k=3) -> list:
        """Recall similar memories based on vector similarity"""
        if not self.vector_index:
            return []
        
        similarities = []
        for timestamp, vector in self.vector_index.items():
            sim = np.dot(query_vector, vector) / (np.linalg.norm(query_vector) * np.linalg.norm(vector))
            similarities.append((timestamp, sim))
        
        similarities.sort(key=lambda x: x[1], reverse=True)
        return [self._find_memory(ts) for ts, _ in similarities[:top_k]]
    
    def _find_memory(self, timestamp):
        """Find memory entry by timestamp"""
        for entry in self.stream:
            if entry['timestamp'] == timestamp:
                return entry
        return None
    
    def save_memory(self):
        with open(self.storage_path, 'w') as f:
            json.dump(list(self.stream), f)
    
    def load_memory(self):
        if os.path.exists(self.storage_path):
            with open(self.storage_path, 'r') as f:
                memories = json.load(f)
                self.stream = deque(memories, maxlen=self.stream.maxlen)
                for entry in memories:
                    self.vector_index[entry['timestamp']] = np.array(entry['vector'])
