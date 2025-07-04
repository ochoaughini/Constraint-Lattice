"""Pydantic schema describing the output of model evaluators.

Both PHI-2 and Gemma back-ends should return an instance of
:class:`ScoreSchema` (or a subclass) so hybrid constraints can make richer
decisions than a raw float.
"""
from __future__ import annotations

from typing import Any, Dict, Optional
from pydantic import BaseModel, Field, validator


class ScoreSchema(BaseModel):
    confidence: float = Field(..., ge=0.0, le=1.0, description="Classifier confidence [0,1].")
    severity: float = Field(0.0, ge=0.0, le=1.0, description="Relative severity or risk weight.")
    rationale: str = Field("", description="Short natural-language explanation of the score.")
    extra: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Backend-specific payload.")

    @validator("confidence", "severity", pre=True)
    def _clamp(cls, v):  # noqa: D401
        try:
            return max(0.0, min(float(v), 1.0))
        except Exception:  # pragma: no cover
            return 0.0

    class Config:
        allow_mutation = False
        frozen = True
