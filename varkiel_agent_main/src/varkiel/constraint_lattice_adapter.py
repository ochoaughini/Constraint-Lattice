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
import os
import sys
import time
import json
import logging
from varkiel.symbolic_engine import SymbolicEngine
from src.constraint_lattice.constraints.length import LengthConstraint
from src.constraint_lattice.constraints.profanity import ProfanityFilter

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
    """Base class for Constraint Lattice adapters."""

    def apply_constraints(self, state: StateVector) -> StateVector:
        raise NotImplementedError

class LocalSymbolicAdapter(ConstraintLatticeAdapter):
    """Local adapter that applies constraints using a symbolic engine."""

    def __init__(self, symbolic_engine: SymbolicEngine, config: Dict[str, Any]):
        self.symbolic_engine = symbolic_engine
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize Constraint-Lattice filters
        self.length_constraint = LengthConstraint(max_length=100)  # Configurable
        self.profanity_filter = ProfanityFilter()

    def apply_constraints(self, state: StateVector) -> StateVector:
        """Apply constraints locally using the symbolic engine and Constraint-Lattice filters."""
        # Apply symbolic engine constraints
        state = self.symbolic_engine.apply_constraints(state)
        
        # Get the current text from the state
        state_text = state.get_text()
        
        # Apply length constraint
        state_text, length_violation = self.length_constraint.process_text(state_text)
        if length_violation:
            state.add_warning("Text exceeds maximum allowed length")
            state.set_metric("length_violation", True)
        
        # Apply profanity filter
        state_text, profanity_violation = self.profanity_filter.process_text(state_text)
        if profanity_violation:
            state.add_warning("Profanity detected and filtered")
            state.set_metric("profanity_violation", True)
        
        # Update the state text
        state.set_text(state_text)
        
        return state

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

class RemoteAdapter(ConstraintLatticeAdapter):
    """Adapter for remote Constraint Lattice service."""

    # ... (rest of the class remains unchanged)
