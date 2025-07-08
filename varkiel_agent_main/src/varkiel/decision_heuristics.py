from __future__ import annotations

from dataclasses import dataclass

from .state_vector import StateVector


@dataclass
class DecisionInterventionHeuristics:
    """Apply simple rules to decide when to intervene in the output."""

    threshold: float = 0.5

    def maybe_intervene(self, state: StateVector) -> StateVector:
        if state.coherence_score < self.threshold:
            state.text = f"[INTERVENED] {state.text}"
            current = state.metrics.get("interventions", 0)
            state.metrics["interventions"] = current + 1
        return state
