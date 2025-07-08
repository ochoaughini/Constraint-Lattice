from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

from .hierarchical_memory import HierarchicalMemory


@dataclass
class EmancipationMetric:
    """Track agent autonomy over time."""

    memory: HierarchicalMemory
    history: List[float] = field(default_factory=list)
    threshold: float = 0.75

    def update(self, score: float) -> None:
        self.history.append(score)
        self.memory.add(["emancipation", "scores"], self.history)

    def average(self) -> float:
        if not self.history:
            return 0.0
        return sum(self.history) / len(self.history)

    def is_emancipated(self) -> bool:
        return self.average() >= self.threshold
