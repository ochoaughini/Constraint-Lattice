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

Core orchestration component - Complete Implementation
"""

import time
import logging
import traceback
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from varkiel.state_vector import StateVector
from varkiel.exceptions import GovernanceError, SafetyViolationError

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
        self._init_components()
        
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
            ('phenomenological', self.components['phenomenological'].track_state)
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
