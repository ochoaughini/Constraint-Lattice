from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List


@dataclass
class Agent:
    name: str
    act: Callable[[str], str]


@dataclass
class GovernanceCoordinator:
    agents: List[Agent] = field(default_factory=list)

    def register(self, agent: Agent) -> None:
        self.agents.append(agent)

    def broadcast(self, message: str) -> Dict[str, str]:
        """Send the same message to all agents and collect their responses."""
        return {agent.name: agent.act(message) for agent in self.agents}

    def consensus(self, message: str) -> str:
        """Return the response that most agents agree on."""
        responses = self.broadcast(message)
        vote_count: Dict[str, int] = {}
        for resp in responses.values():
            vote_count[resp] = vote_count.get(resp, 0) + 1
        return max(vote_count, key=vote_count.get)
