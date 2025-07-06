# SPDX-License-Identifier: MIT
# Copyright (c) 2025 ochoaughini. See LICENSE for full terms.
# Copyright (c) 2025 ochoaughini. See LICENSE for full terms.
# Copyright (c) 2025 ochoaughini. See LICENSE for full terms.
"""
Varkiel Agent - Advanced AI Constraint System
Copyright (C) 2025 Lexsight LLC
SPDX-License-Identifier: AGPL-3.0-only OR Commercial

Core orchestration component - Complete Implementation
"""

import time
import logging
import traceback
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from varkiel.state_vector import StateVector
from varkiel.exceptions import GovernanceError, SafetyViolationError
from wildcore.detector import AutoRegulatedPromptDetector
from sentence_transformers import SentenceTransformer
from varkiel.policy_dsl import PolicyEngine
from varkiel.coherence import RecursiveInvarianceMonitor
import numpy as np
import pickle

@dataclass
class ProcessingResult:
    output: str
    metrics: Dict[str, float]
    warnings: List[str]
    audit_trail: Dict[str, Any]
    processing_time: float

class CentralController:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.reference_embeddings = self._load_reference_embeddings()
        self._init_components()
        self.policy_engine = PolicyEngine()
        self.coherence_monitor = RecursiveInvarianceMonitor()
        
    def _init_components(self):
        """Initialize all processing components"""
        from varkiel.structural_constraint_engine import StructuralEngine
        from varkiel.symbolic_coherence_engine import SymbolicEngine
        from varkiel.phenomenological_tracker import PhenomenologicalTracker
        from varkiel.risk_balancer import RiskBalancer
        from varkiel.audit_logger import AuditLogger
        
        self.components = {
            'structural': StructuralEngine(self.config.get('structural', {})),
            'symbolic': SymbolicEngine(self.config.get('symbolic', {})),
            'phenomenological': PhenomenologicalTracker(self.config.get('phenomenological', {})),
            'risk': RiskBalancer(self.config.get('risk', {})),
            'audit': AuditLogger(self.config.get('audit', {}))
        }
        
        if self.config.get('use_lattice', False):
            from varkiel.constraint_lattice_adapter import ConstraintLatticeAdapter
            self.components['lattice'] = ConstraintLatticeAdapter(
                self.config['lattice']['endpoint'],
                self.config['lattice']['api_key']
            )
        
        # Initialize WildCore components
        self.wildcore_detector = AutoRegulatedPromptDetector(
            reference_embeddings=self.reference_embeddings,
            threshold=self.config['wildcore']['threshold']
        )
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
    def _load_reference_embeddings(self) -> List[np.ndarray]:
        """Load reference embeddings from file"""
        embeddings_path = self.config['wildcore']['embeddings_path']
        try:
            with open(embeddings_path, 'rb') as f:
                return pickle.load(f)
        except FileNotFoundError:
            self.logger.warning(f"Embeddings file not found at {embeddings_path}")
            return []
        
    def process(self, input_text: str) -> ProcessingResult:
        """Full processing pipeline with timing and error handling"""
        start_time = time.time()
        state = StateVector.from_input(input_text)
        
        try:
            # Core processing pipeline
            state = self._apply_processing_pipeline(state)
            
            # Governance layer if configured
            if 'lattice' in self.components:
                state = self._apply_governance(state)
            
            # Final safety check
            if not self.components['risk'].approve(state):
                violation_details = self.components['risk'].get_violation_details(state)
                raise SafetyViolationError("Output failed safety checks", details=violation_details)
                
            # Evaluate policies
            policy_result = self.policy_engine.evaluate(state)
            if policy_result.violations:
                state.warnings.append(f"Policy violations: {policy_result.violations}")
                
            # Apply coherence monitoring
            state = self._apply_coherence_monitoring(state)
            
            return ProcessingResult(
                output=state.text,
                metrics=state.metrics,
                warnings=state.warnings,
                audit_trail=state.audit_data,
                processing_time=time.time() - start_time
            )
            
        except SafetyViolationError as e:
            self.logger.error(f"Safety violation details: {e.details}")
            raise
        except Exception as e:
            context = {
                'input_text': input_text,
                'state': state.to_dict() if state else None,
                'exception': str(e),
                'stack_trace': traceback.format_exc()
            }
            self.components['audit'].log_error(e, context)
            self.logger.exception("Processing failed")
            raise

    def _apply_processing_pipeline(self, state: StateVector) -> StateVector:
        """Apply all core processing layers"""
        processing_order = [
            ('structural', self.components['structural'].apply_constraints),
            ('symbolic', self.components['symbolic'].process),
            ('phenomenological', self.components['phenomenological'].track_state),
            ('wildcore', self._apply_wildcore_detection)  # New step for WildCore
        ]
        
        for stage_name, processor in processing_order:
            try:
                stage_start = time.time()
                state = processor(state)
                state.metrics[f'{stage_name}_time'] = time.time() - stage_start
            except Exception as e:
                self.logger.error(f"{stage_name} processing failed: {str(e)}")
                state.warnings.append(f"{stage_name} processing failed")
                raise
                
        return state
        
    def _apply_wildcore_detection(self, state: StateVector) -> StateVector:
        """Apply WildCore anomaly detection"""
        try:
            # Generate embedding for the current state text
            embedding = self.embedding_model.encode(state.text)
            # TODO: Load reference embeddings from config or a file
            result = self.wildcore_detector.ensemble_detection(embedding)
            if result['is_anomalous']:
                state.warnings.append(f"WildCore anomaly detected: {result.get('methods_triggered', [])}")
                state.metrics['wildcore_anomaly'] = True
            else:
                state.metrics['wildcore_anomaly'] = False
        except Exception as e:
            self.logger.error(f"WildCore detection failed: {str(e)}")
            state.warnings.append("WildCore detection failed")
        return state

    def _apply_coherence_monitoring(self, state: StateVector) -> StateVector:
        """Apply recursive invariance monitoring for semantic coherence"""
        try:
            state = self.coherence_monitor.monitor(state)
        except Exception as e:
            self.logger.error(f"Coherence monitoring failed: {e}")
            state.add_warning("Coherence monitoring failed")
        return state

    def _apply_governance(self, state: StateVector) -> StateVector:
        """Apply governance constraints via Lattice adapter"""
        try:
            governance_result = self.components['lattice'].govern(
                state.text,
                self.config['lattice'].get('profile', 'default')
            )
            state.text = governance_result['output']
            state.audit_data['governance'] = governance_result['audit_trail']
            state.metrics['governance_time'] = governance_result['processing_time']
        except Exception as e:
            self.logger.error(f"Governance failed: {str(e)}")
            state.warnings.append("Governance processing failed")
            raise GovernanceError(f"Governance processing failed: {str(e)}")
            
        return state
