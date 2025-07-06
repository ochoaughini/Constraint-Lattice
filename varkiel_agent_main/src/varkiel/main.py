# SPDX-License-Identifier: MIT
# Copyright (c) 2025 ochoaughini. See LICENSE for full terms.
# Copyright (c) 2025 ochoaughini. See LICENSE for full terms.
# Copyright (c) 2025 ochoaughini. See LICENSE for full terms.
from central_controller import CentralController
from structural_constraint_engine import StructuralConstraintEngine
from constraint_lattice_adapter import ConstraintLatticeAdapter
from semantic_resonance import FormStateVector

if __name__ == "__main__":
    # Initialize components
    constraint_lattice = ConstraintLatticeAdapter()
    structural_engine = StructuralConstraintEngine(constraint_lattice)
    
    # Initialize controller
    controller = CentralController(
        structural_engine=structural_engine,
        coherence_engine=structural_engine,  # Using same engine for simplicity
        phenomenological_tracker=FormStateVector(),
        recursive_invariance_monitor=structural_engine
    )
    
    # Run interactive session
    while True:
        prompt = input("User: ")
        if prompt.lower() in ['exit', 'quit']:
            break
        response = controller.process_query(prompt)
        print(f"System: {response}")
