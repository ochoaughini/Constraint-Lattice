"""Tenant-level policy definition for Constraint-Lattice SaaS layer.

A TenantPolicy bundles:
• constraints – instantiated Constraint objects
• evaluator chain preferences
• rate-limits (requests per minute, etc.)
• RBAC scopes (FastAPI dependency can enforce)

The FastAPI SaaS layer is expected to resolve the tenant from the request,
fetch or mint a TenantPolicy (e.g. from Postgres or Redis), and hand it to
ConstraintEngine so per-request instances are fully configured.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Sequence

from engine.evaluators import ModelEvaluator, FallbackEvaluator
from engine.scheduler import Constraint

__all__ = ["TenantPolicy"]


@dataclass(slots=True)
class TenantPolicy:
    tenant_id: str
    constraints: Sequence[Constraint] = field(default_factory=list)
    evaluators: List[ModelEvaluator] = field(default_factory=lambda: [FallbackEvaluator()])
    rps_limit: int = 60  # requests per minute
    scopes: frozenset[str] = frozenset({"default"})

    def with_constraint_overrides(self, new_constraints: Sequence[Constraint]) -> "TenantPolicy":
        return TenantPolicy(
            tenant_id=self.tenant_id,
            constraints=new_constraints,
            evaluators=self.evaluators,
            rps_limit=self.rps_limit,
            scopes=self.scopes,
        )
