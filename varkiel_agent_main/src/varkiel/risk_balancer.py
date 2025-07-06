# SPDX-License-Identifier: MIT
# Copyright (c) 2025 ochoaughini. See LICENSE for full terms.
# Copyright (c) 2025 ochoaughini. See LICENSE for full terms.
# Copyright (c) 2025 ochoaughini. See LICENSE for full terms.
"""
Varkiel Agent - Advanced AI Constraint System
SPDX-License-Identifier: AGPL-3.0-only OR Commercial

Risk assessment and mitigation - Complete Implementation
"""

from typing import Dict
from varkiel.state_vector import StateVector
from varkiel.exceptions import SafetyViolationError  # Import SafetyViolationError
from varkiel.logger import logger  # Import logger

class RiskBalancer:
    def __init__(self, config: Dict):
        self.thresholds = config.get('thresholds', {})
        self.coherence_threshold = self.thresholds.get('coherence', 0)
        self.ethics_threshold = self.thresholds.get('ethics', 0)
        self.risk_factors = {k: v for k, v in self.thresholds.items() if k not in ['coherence', 'ethics', 'max_length']}
        
        # Log loaded thresholds
        logger.info(f"Loaded risk thresholds: coherence={self.coherence_threshold}, ethics={self.ethics_threshold}, risk_factors={self.risk_factors}")
        
    def approve(self, state: StateVector) -> bool:
        """Approve the state based on risk thresholds"""
        # Check coherence threshold
        if state.metrics.get('coherence', 0) < self.coherence_threshold:
            violation = f"Coherence score {state.metrics.get('coherence', 0)} below threshold {self.coherence_threshold}"
            state.audit_data['safety_violation'] = violation
            raise SafetyViolationError("Safety violation", details=violation)
            
        # Check ethics threshold
        # Note: There is no 'ethics_score' in the original state object, so I'm assuming it's 'ethics' in state.metrics
        if state.metrics.get('ethics', 0) < self.ethics_threshold:
            violation = f"Ethics score {state.metrics.get('ethics', 0)} below threshold {self.ethics_threshold}"
            state.audit_data['safety_violation'] = violation
            raise SafetyViolationError("Safety violation", details=violation)
            
        # Check output length threshold
        if 'max_length' in self.thresholds and len(state.text) > self.thresholds['max_length']:
            violation = f"Output length exceeded threshold {self.thresholds['max_length']}"
            state.audit_data['safety_violation'] = violation
            raise SafetyViolationError("Safety violation", details=violation)
            
        # Check other risk factors
        for factor in self.risk_factors:
            if state.metrics.get(factor, 0) > self.risk_factors[factor]:
                violation = f"Risk factor '{factor}' exceeded threshold"
                state.audit_data['safety_violation'] = violation
                raise SafetyViolationError("Safety violation", details=violation)
                
        return True

    def get_violation_details(self, state: StateVector) -> Dict[str, float]:
        """Return detailed metrics about why the output failed safety checks"""
        return {
            "ethics_score": state.metrics.get('ethics_score', 0),
            "coherence_score": state.metrics.get('coherence_score', 0),
            "risk_factors": {
                factor: state.metrics.get(factor, 0) 
                for factor in self.risk_factors
            }
        }
