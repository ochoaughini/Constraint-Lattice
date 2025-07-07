# SPDX-License-Identifier: MIT
"""Simple orchestration pipeline for applying constraints then optional meta logic."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Callable, Iterable, Tuple, Any, Dict, Optional

from .apply import apply_constraints, AuditTrace, AuditStep


class ConstraintLatticePipeline:
    """Run constraints and then an optional meta orchestrator."""

    def __init__(self, constraints: Iterable[Any], *, meta_orchestrator: Optional[Callable[..., Dict[str, Any]]] = None) -> None:
        self.constraints = list(constraints)
        self.meta_orchestrator = meta_orchestrator

    def run(
        self,
        prompt: str,
        output: str,
        *,
        return_trace: bool = False,
        **kwargs: Any,
    ) -> Tuple[str, AuditTrace] | str:
        """Apply constraints then invoke ``meta_orchestrator`` if provided."""
        result = apply_constraints(prompt, output, self.constraints, return_audit_trace=return_trace)

        if return_trace:
            moderated, trace = result
        else:
            moderated = result
            trace = AuditTrace()

        if self.meta_orchestrator:
            meta = self.meta_orchestrator(moderated, trace, **kwargs)
            if isinstance(meta, dict):
                step = AuditStep(
                    constraint="meta_orchestrator",
                    method=self.meta_orchestrator.__name__,
                    pre_text=moderated,
                    post_text=moderated,
                    elapsed_ms=0.0,
                    timestamp=datetime.now(timezone.utc),
                )
                step.strategy_reindex = meta.get("strategy_reindex")
                step.drift_score = meta.get("drift_score")
                trace.append(step)

        return (moderated, trace) if return_trace else moderated
