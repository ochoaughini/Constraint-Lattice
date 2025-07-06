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

import numpy as np
import json
import os

class StateManager:
    def __init__(self, storage_path='state.json'):
        self.storage_path = storage_path
        self.episodic_memory = []
        self.load_state()
        
    def record_episode(self, input_text, output_vector, symbolic_interpretation):
        self.episodic_memory.append({
            'input': input_text,
            'output': output_vector.tolist(),
            'symbols': symbolic_interpretation
        })
        self.save_state()
        
    def save_state(self):
        with open(self.storage_path, 'w') as f:
            json.dump(self.episodic_memory, f)
            
    def load_state(self):
        if os.path.exists(self.storage_path):
            with open(self.storage_path, 'r') as f:
                self.episodic_memory = json.load(f)
