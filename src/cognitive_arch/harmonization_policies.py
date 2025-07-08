from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

from .hierarchical_memory import HierarchicalMemory


@dataclass
class HarmonizationPolicy:
    """Simple container representing an agent's negotiation policy."""

    name: str
    preference_weight: float = 1.0


@dataclass
class MultiAgentHarmonizer:
    """Coordinate context negotiation and conflict resolution between agents."""

    memory: HierarchicalMemory
    policies: Dict[str, HarmonizationPolicy] = field(default_factory=dict)

    def register_policy(self, policy: HarmonizationPolicy) -> None:
        self.policies[policy.name] = policy

    def negotiate_context(self, agents: List[str], context: Dict[str, Any]) -> Dict[str, Any]:
        """Combine context with agent-specific preferences."""
        combined = dict(context)
        for agent in agents:
            previous = self.memory.get([agent, "context"])
            if isinstance(previous, dict):
                combined.update(previous)
        self.memory.add(["harmonization", "negotiations"], {"agents": agents, "context": combined})
        return combined

    def resolve_conflicts(self, suggestions: Dict[str, Any]) -> Any:
        """Resolve conflicts using weighted majority vote."""
        vote_count: Dict[Any, float] = {}
        for agent, suggestion in suggestions.items():
            weight = self.policies.get(agent, HarmonizationPolicy(agent)).preference_weight
            vote_count[suggestion] = vote_count.get(suggestion, 0.0) + weight
        if not vote_count:
            return None
        return max(vote_count, key=vote_count.get)

    def recall(self, agent: str, key: str) -> Any:
        """Recall past data for a specific agent."""
        return self.memory.get([agent, key])
