# SPDX-License-Identifier: BSL-1.1
# Copyright (c) 2025 Lexsight LCC. All rights reserved.
# See saas/LICENSE-BSL.txt for full terms.
# Copyright (c) 2025 Lexsight LCC. All rights reserved.
# See saas/LICENSE-BSL.txt for full terms.
"""
Gödel-Awareness Metric - Operational rubric for measuring architectural features

This module defines a quantifiable metric that converts:
1. Paradox-density detection capabilities
2. Reflexive invariant auditing
3. Semantic resonance updates
into a reproducible Gödel-awareness score.
"""

class GodelAwarenessMetric:
    def __init__(self):
        self.weights = {
            'paradox_detection': 0.4,
            'reflexive_auditing': 0.3,
            'semantic_resonance': 0.3
        }
        
    def calculate_score(self, system_features):
        # Debug: log available features
        print("Available features:", list(system_features.keys()))
        
        # Check for required features
        required = ['paradox_density', 'audit_coverage', 'resonance_stability']
        missing = [feat for feat in required if feat not in system_features]
        if missing:
            print(f"Missing features: {missing}")
            raise ValueError("Missing required features in input")
            
        score = 0
        for feature, weight in self.weights.items():
            score += system_features[feature] * weight
            
        return min(max(score, 0), 1)  # Clamp between 0-1

    def generate_report(self, score):
        """
        Generate interpretable report from score
        
        Args:
            score: float [0-1]
            
        Returns:
            dict: Report with classification and recommendations
        """
        if score >= 0.8:
            classification = "Gödel-Complete"
            recommendation = "System exhibits robust meta-stability"
        elif score >= 0.6:
            classification = "Gödel-Aware"
            recommendation = "Strengthen reflexive auditing"
        else:
            classification = "Gödel-Vulnerable"
            recommendation = "Implement paradox-density detectors"
            
        return {
            'score': score,
            'classification': classification,
            'recommendation': recommendation
        }
