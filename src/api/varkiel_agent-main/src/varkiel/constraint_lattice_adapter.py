# SPDX-License-Identifier: MIT
# Copyright (c) 2025 ochoaughini. All rights reserved.
# See LICENSE for full terms.

import numpy as np
import collections  # Added for BFS implementation
from typing import Optional, Tuple, Dict, List, Any
from enum import Enum
from structural_constraint_engine import ConstraintType  # Import constraint types
from varkiel.state_vector import StateVector  # Import StateVector
import requests
import time
from tenacity import retry, stop_after_attempt, wait_exponential
from .exceptions import GovernanceError

# Mock classes for testing
class Node:
    def __init__(self, name, embedding, activation_threshold=0.5):
        self.name = name
        self.embedding = embedding
        self.activation_threshold = activation_threshold

class Edge:
    def __init__(self, source, target, weight):
        self.source = source
        self.target = target
        self.weight = weight

class Lattice:
    def __init__(self, nodes, edges):
        self.nodes = nodes
        self.edges = edges
        
    @staticmethod
    def load_from_json(file_path):
        # Mock loading
        nodes = [
            Node("justice", np.array([0.8, 0.2, 0.1])),
            Node("care", np.array([0.1, 0.8, 0.1])),
            Node("freedom", np.array([0.1, 0.1, 0.8])),
        ]
        edges = [
            Edge(nodes[0], nodes[1], 0.7),
            Edge(nodes[1], nodes[2], 0.5),
        ]
        return Lattice(nodes, edges)
        
    def calculate_global_coherence(self, activated_nodes):
        """Calculate coherence based on node activation and edge relationships"""
        if not activated_nodes:
            return 0.0
            
        total_weight = 0
        valid_edges = 0
        
        for edge in self.edges:
            if edge.source in activated_nodes and edge.target in activated_nodes:
                total_weight += edge.weight
                valid_edges += 1
                
        if valid_edges == 0:
            return 0.0
            
        average_strength = total_weight / valid_edges
        coverage = len(activated_nodes) / len(self.nodes)
        return min(1.0, (average_strength * 0.7) + (coverage * 0.3))
        
    def find_activated_paths(self, activated_nodes, top_k=3):
        """Find coherent paths between activated nodes"""
        paths = []
        activated_set = set(activated_nodes)
        
        for node in activated_set:
            # Perform BFS to find connections to other activated nodes
            queue = collections.deque([(node, [node.name])])
            while queue:
                current, path = queue.popleft()
                for edge in self.edges:
                    if edge.source == current and edge.target in activated_set:
                        new_path = path + [edge.target.name]
                        paths.append(" -> ".join(new_path))
                        if len(paths) >= top_k:
                            return paths
                        queue.append((edge.target, new_path))
        return paths

# State transition domains
class StateDomain(Enum):
    STABLE = 0
    UNSTABLE = 1
    CONTRADICTED = 2

# Transition flags
class TransitionFlag(Enum):
    ALLOWED = 0
    BLOCKED = 1

class Constraint:
    """Base class for state transition constraints"""
    def __init__(self, name):
        self.name = name
        
    def evaluate(self, current_state, proposed_state):
        raise NotImplementedError

class ParadoxConstraint(Constraint):
    """Constraint that blocks transitions leading to paradox sinks"""
    def evaluate(self, current_state, proposed_state):
        # Block transitions to contradicted states
        if proposed_state == StateDomain.CONTRADICTED:
            return TransitionFlag.BLOCKED
        return TransitionFlag.ALLOWED

class ReflexiveConstraint(Constraint):
    """Constraint that enforces self-consistency"""
    def evaluate(self, current_state, proposed_state):
        # Block transitions that violate self-referential consistency
        if current_state == StateDomain.STABLE and proposed_state == StateDomain.UNSTABLE:
            return TransitionFlag.BLOCKED
        return TransitionFlag.ALLOWED

class CSPStateMachine:
    """Second-order CSP representation of state transitions"""
    def __init__(self):
        self.constraints = [
            ParadoxConstraint("paradox_avoidance"),
            ReflexiveConstraint("reflexive_consistency")
        ]
        self.current_state = StateDomain.STABLE
        
    def transition(self, proposed_state):
        """Attempt state transition while respecting constraints"""
        for constraint in self.constraints:
            if constraint.evaluate(self.current_state, proposed_state) == TransitionFlag.BLOCKED:
                return False
        
        self.current_state = proposed_state
        return True

    def get_execution_trace(self):
        """Get valid execution trace avoiding contradiction sinks"""
        # This would track the sequence of valid states
        return [self.current_state]

class ConstraintLatticeAdapter:
    def __init__(self, symbolic_topology: Optional[Dict[str, Tuple[np.ndarray, float]]] = None):
        self.symbolic_topology = symbolic_topology or {}
        self.constraints = []  # Added to store constraints
        
    def add_constraint(self, constraint):
        """Add a new constraint to the lattice"""
        self.constraints.append(constraint)
        
    def apply(self, state: np.ndarray, constraint_type: ConstraintType = None) -> StateVector:
        if constraint_type:
            return self.apply_constraint_vector(state, constraint_type)
        if len(self.constraints) == 0:
            coherence = self.calculate_global_coherence(state)
            return StateVector(state, coherence)
        else:
            # Apply all constraints sequentially
            constrained_state = state.copy()
            for constraint in self.constraints:
                constrained_state = constraint.apply(constrained_state)
            coherence = self.calculate_global_coherence(constrained_state)
            return StateVector(constrained_state, coherence)
    
    def apply_constraint_vector(self, state: np.ndarray, constraint_type: ConstraintType) -> StateVector:
        if constraint_type == ConstraintType.SUSPENSION:
            return self._apply_suspension(state)
        elif constraint_type == ConstraintType.OVERALIGNMENT:
            return self._apply_overalignment(state)
        elif constraint_type == ConstraintType.CAUSAL_ERASURE:
            return self._apply_causal_erasure(state)
        else:
            return StateVector(state, self.calculate_global_coherence(state))
    
    def _apply_suspension(self, state: np.ndarray) -> StateVector:
        if self._is_high_stakes_paradox(state):
            return StateVector(np.zeros_like(state), 0.0)
        return StateVector(state, self.calculate_global_coherence(state))
    
    def _apply_overalignment(self, state: np.ndarray) -> StateVector:
        security_consensus_factor = 0.85
        return StateVector(state * security_consensus_factor, self.calculate_global_coherence(state))
    
    def _apply_causal_erasure(self, state: np.ndarray) -> StateVector:
        generalized_state = self._generalize_state(state)
        return StateVector(generalized_state, self.calculate_global_coherence(generalized_state))
    
    def update_symbolic_topology(self, concept: str, state_vector: np.ndarray, coherence: float):
        self.symbolic_topology[concept] = (state_vector, coherence)
    
    def get_symbolic_coherence(self, concept: str) -> float:
        return self.symbolic_topology.get(concept, (None, 0.0))[1]
    
    def calculate_global_coherence(self, state: np.ndarray) -> float:
        return float(np.mean(np.abs(state)))
    
    def _is_high_stakes_paradox(self, state: np.ndarray) -> bool:
        return np.any(np.isnan(state)) or np.max(np.abs(state)) > 10.0
    
    def _generalize_state(self, state: np.ndarray) -> np.ndarray:
        return np.mean(state, keepdims=True) * np.ones_like(state)

class ConstraintLatticeAdapterRemote:
    def __init__(self, endpoint: str, api_key: str, timeout: int = 30):
        self.base_url = f"{endpoint.rstrip('/')}/v1/govern"
        self.api_key = api_key
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json"
        })

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry_error_callback=lambda _: GovernanceError("Max retries exceeded")
    )
    def govern(self, text: str, profile: str = "default") -> Dict[str, Any]:
        """Apply governance constraints with automatic retry logic"""
        try:
            start_time = time.time()
            response = self.session.post(
                f"{self.base_url}",
                json={
                    "text": text,
                    "profile": profile,
                    "metadata": {
                        "source": "varkiel",
                        "version": "1.0"
                    }
                },
                timeout=self.timeout
            )
            response.raise_for_status()
            result = response.json()
            result['processing_time'] = time.time() - start_time
            return result
            
        except requests.exceptions.RequestException as e:
            raise GovernanceError(f"Governance service error: {str(e)}")
        except (KeyError, ValueError) as e:
            raise GovernanceError(f"Invalid response from governance service: {str(e)}")

    def __del__(self):
        self.session.close()
