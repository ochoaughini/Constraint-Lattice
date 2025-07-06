# SPDX-License-Identifier: BSL-1.1
# Copyright (c) 2025 Lexsight LCC. All rights reserved.
# See saas/LICENSE-BSL.txt for full terms.
# Copyright (c) 2025 Lexsight LCC. All rights reserved.
# See saas/LICENSE-BSL.txt for full terms.
"""
Varkiel Agent - Advanced AI Constraint System
SPDX-License-Identifier: AGPL-3.0-only OR Commercial

Custom exceptions for the Varkiel system.
"""


class GovernanceError(Exception):
    """Exception raised for governance constraint violations."""
    pass


class SafetyViolationError(Exception):
    """Exception raised when safety thresholds are violated."""
    def __init__(self, message, details=None):
        super().__init__(message)
        self.details = details or message


class CoherenceError(Exception):
    """Exception raised for coherence violations."""
    pass


class StructuralConstraintError(Exception):
    """Exception raised when structural constraints are violated."""
    pass
