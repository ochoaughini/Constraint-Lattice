from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Dict


@dataclass
class JustificationStep:
    node_id: str
    reason: str


@dataclass
class ChainOfJustification:
    steps: List[JustificationStep] = field(default_factory=list)

    def add_step(self, node_id: str, reason: str) -> None:
        self.steps.append(JustificationStep(node_id, reason))

    def to_dict(self) -> Dict[str, List[Dict[str, str]]]:
        return {"chain": [step.__dict__ for step in self.steps]}

