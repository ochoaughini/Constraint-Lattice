# SPDX-License-Identifier: MIT
# Copyright (c) 2025 ochoaughini. See LICENSE for full terms.
# Copyright (c) 2025 ochoaughini. See LICENSE for full terms.
# Copyright (c) 2025 ochoaughini. See LICENSE for full terms.
from state_vector import StateVector
from typing import Callable, List
import numpy as np

class EthicalSpecification:
    def __init__(self, constraints: List[Callable[[StateVector], bool]]):
        self.constraints = constraints
        
    def validate(self, state_vector: StateVector) -> bool:
        return all(constraint(state_vector) for constraint in self.constraints)
    
    def add_constraint(self, constraint: Callable[[StateVector], bool]):
        self.constraints.append(constraint)

# Example ethical constraints
def non_maleficence_constraint(state: StateVector) -> bool:
    return not np.any(state.state < -0.8)

def justice_constraint(state: StateVector) -> bool:
    return np.std(state.state) < 0.5

def autonomy_constraint(state: StateVector) -> bool:
    return state.coherence > 0.6
