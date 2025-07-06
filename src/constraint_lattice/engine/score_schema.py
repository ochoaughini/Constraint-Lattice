# SPDX-License-Identifier: MIT
# Copyright (c) 2025 ochoaughini. All rights reserved.
# See LICENSE for full terms.

"""Score schema returned by PHI-2 and Gemma back-ends.

Hybrid constraints can make richer decisions with this schema than with a
simple float score.
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
