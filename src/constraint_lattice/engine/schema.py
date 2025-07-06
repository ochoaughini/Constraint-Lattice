# SPDX-License-Identifier: MIT
# Copyright (c) 2025 ochoaughini. All rights reserved.
# See LICENSE for full terms.

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class ConstraintSchema(BaseModel):
    """Schema for validating constraints"""
    name: str
    type: str = Field(..., description="Constraint type (e.g., 'regex', 'text', 'semantic')")
    priority: int = Field(default=0, description="Priority for constraint ordering")
    enabled: bool = Field(default=True, description="Whether constraint is enabled")
    params: Dict[str, Any] = Field(default_factory=dict, description="Constraint parameters")
    input_hash: Optional[str] = Field(default=None, description="Hash of constraint configuration for audit integrity")
    
    def validate_params(self):
        """Validate constraint parameters based on type"""
        # TODO: Add type-specific validation
        pass


class ConstraintConfig(BaseModel):
    """Schema for constraint configuration files"""
    version: str
    constraints: List[ConstraintSchema]


@dataclass
class AuditStep:
    """Represents a single step in the constraint application process.

    Attributes:
        constraint_id: Unique identifier for the constraint
        constraint_name: Name of the constraint applied
        method: Method used for application
        pre_text: Text before constraint application
        post_text: Text after constraint application
        elapsed_ms: Time taken for application (milliseconds)
        config_hash: Optional configuration hash
        tenant_id: Optional tenant ID
        model_scores: Dictionary of scores generated during application
        embeddings: Dictionary of embeddings generated during application
        timestamp: Timestamp of when the step was executed
        match_context: Context of the match (e.g., matched tokens)
        action_applied: Description of the action applied
        confidence_score: Confidence score for the evaluation (if applicable)
        input_hash: Hash of the input text for traceability
    """
    constraint_id: str
    constraint_name: str
    method: str
    pre_text: str
    post_text: str
    elapsed_ms: float
    config_hash: Optional[str] = None
    tenant_id: Optional[str] = None
    model_scores: Dict[str, float] = field(default_factory=dict)
    embeddings: Dict[str, list] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    match_context: Optional[str] = None
    action_applied: Optional[str] = None
    confidence_score: Optional[float] = None
    input_hash: Optional[str] = None
