import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "varkiel_agent_main" / "src"))

from cognitive_arch.harmonization_policies import MultiAgentHarmonizer, HarmonizationPolicy
from cognitive_arch.hierarchical_memory import HierarchicalMemory
from dataclasses import dataclass, field


class DecisionInterventionHeuristics:
    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold

    def maybe_intervene(self, state: "StateVector") -> "StateVector":
        if state.coherence_score < self.threshold:
            state.text = f"[INTERVENED] {state.text}"
            state.metrics["interventions"] = state.metrics.get("interventions", 0) + 1
        return state


@dataclass
class StateVector:
    text: str
    coherence_score: float = 0.0
    metrics: dict[str, float] = field(default_factory=dict)
from cognitive_arch.emancipation_metric import EmancipationMetric


def test_multi_agent_conflict_resolution():
    mem = HierarchicalMemory()
    harmonizer = MultiAgentHarmonizer(mem)
    harmonizer.register_policy(HarmonizationPolicy("a", 2.0))
    choice = harmonizer.resolve_conflicts({"a": "yes", "b": "no", "c": "no"})
    assert choice == "yes"


def test_decision_intervention():
    heur = DecisionInterventionHeuristics(threshold=0.5)
    state = StateVector(text="hello", coherence_score=0.3)
    new_state = heur.maybe_intervene(state)
    assert new_state.text.startswith("[INTERVENED]")
    assert new_state.metrics["interventions"] == 1


def test_emancipation_metric():
    mem = HierarchicalMemory()
    metric = EmancipationMetric(mem, threshold=0.7)
    metric.update(0.8)
    metric.update(0.4)
    assert metric.average() == (0.8 + 0.4) / 2
    assert not metric.is_emancipated()
