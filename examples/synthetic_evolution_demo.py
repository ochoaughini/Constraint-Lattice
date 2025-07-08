"""Demonstration of synthetic cognitive evolution across LLM agents.

This script wires together the lightweight cognitive modules to show how
agents might negotiate context, share memory and track autonomy.
"""
from cognitive_arch.harmonization_policies import (
    MultiAgentHarmonizer,
    HarmonizationPolicy,
)
from cognitive_arch.hierarchical_memory import HierarchicalMemory
from cognitive_arch.emancipation_metric import EmancipationMetric


class CentralController:
    """Minimal controller integrating harmonization and autonomy tracking."""

    def __init__(self) -> None:
        self.memory = HierarchicalMemory()
        self.harmonizer = MultiAgentHarmonizer(self.memory)
        self.emancipation = EmancipationMetric(self.memory)

    def register_agent(self, name: str, weight: float = 1.0) -> None:
        self.harmonizer.register_policy(HarmonizationPolicy(name, weight))

    def decide(self, proposals: dict[str, str]) -> str | None:
        choice = self.harmonizer.resolve_conflicts(proposals)
        # simplistic autonomy score: 1 when a decision is made
        self.emancipation.update(1.0 if choice else 0.0)
        return choice


if __name__ == "__main__":
    controller = CentralController()
    controller.register_agent("agent_a", 1.5)
    controller.register_agent("agent_b", 1.0)

    proposals = {"agent_a": "acao1", "agent_b": "acao2"}
    decision = controller.decide(proposals)
    print("Decisao escolhida:", decision)
    print("Autonomia media:", controller.emancipation.average())
