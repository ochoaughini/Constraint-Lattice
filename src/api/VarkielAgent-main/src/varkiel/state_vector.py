# SPDX-License-Identifier: BSL-1.1
# Copyright (c) 2025 Lexsight LCC. All rights reserved.
# See saas/LICENSE-BSL.txt for full terms.
# Copyright (c) 2025 Lexsight LCC. All rights reserved.
# See saas/LICENSE-BSL.txt for full terms.
"""
Varkiel Agent - Advanced AI Constraint System
SPDX-License-Identifier: AGPL-3.0-only OR Commercial

State vector representation - Complete Implementation
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List
import numpy as np
from datetime import datetime

@dataclass
class StateVector:
    text: str
    embeddings: np.ndarray = field(default_factory=lambda: np.array([]))
    original_embeddings: np.ndarray = field(default_factory=lambda: np.array([]))
    coherence_score: float = 0.0
    metrics: Dict[str, float] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    audit_data: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    
    @classmethod
    def from_input(cls, text: str):
        """Create initial state from input text"""
        return cls(text=text, metrics={'input_length': len(text)})
    
    def add_metric(self, name: str, value: float):
        """Add processing metric"""
        self.metrics[name] = value
        
    def add_warning(self, message: str):
        """Add processing warning"""
        self.warnings.append(message)
        
    def add_audit_event(self, event_type: str, data: Dict[str, Any]):
        """Add audit trail entry"""
        if 'events' not in self.audit_data:
            self.audit_data['events'] = []
        self.audit_data['events'].append({
            'type': event_type,
            'timestamp': self.timestamp,
            'data': data
        })

    def to_dict(self) -> dict:
        """Convert StateVector to a dictionary"""
        return {
            'text': self.text,
            'embeddings': self.embeddings.tolist() if self.embeddings is not None else None,
            'original_embeddings': self.original_embeddings.tolist() if self.original_embeddings is not None else None,
            'coherence_score': self.coherence_score,
            'metrics': self.metrics,
            'warnings': self.warnings,
            'audit_data': self.audit_data,
            'timestamp': self.timestamp
        }
