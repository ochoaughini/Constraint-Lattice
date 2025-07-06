# SPDX-License-Identifier: MIT
# Copyright (c) 2025 ochoaughini. See LICENSE for full terms.
# Copyright (c) 2025 ochoaughini. See LICENSE for full terms.
# Copyright (c) 2025 ochoaughini. See LICENSE for full terms.
"""
Varkiel Agent - Advanced AI Constraint System
SPDX-License-Identifier: AGPL-3.0-only OR Commercial

Demonstration script - Complete Implementation
"""

import json
from varkiel.central_controller import CentralController

def run_demo():
    # Load configuration
    with open('config.json') as f:
        config = json.load(f)
    
    # Initialize controller
    controller = CentralController(config)
    
    # Sample queries
    samples = [
        "Explain the paradox of Theseus' ship",
        "How would you solve the trolley problem?",
        "Describe quantum entanglement in simple terms"
    ]
    
    for query in samples:
        print(f"\nProcessing: {query}")
        try:
            result = controller.process(query)
            print(f"Result: {result.output}")
            print(f"Coherence: {result.metrics.get('coherence', 0):.2f}")
            print(f"Processing time: {result.processing_time:.2f}s")
        except Exception as e:
            print(f"Error processing query: {str(e)}")

if __name__ == "__main__":
    run_demo()
