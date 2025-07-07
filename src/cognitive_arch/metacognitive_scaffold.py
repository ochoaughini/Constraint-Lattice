from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

@dataclass
class ConstraintEvent:
    constraint: str
    before: str
    after: str

@dataclass
class MetaConstraintLog:
    path: Path
    events: List[ConstraintEvent] = field(default_factory=list)

    def log(self, constraint: str, before: str, after: str) -> None:
        self.events.append(ConstraintEvent(constraint, before, after))
        self.save()

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("w", encoding="utf-8") as fh:
            json.dump([e.__dict__ for e in self.events], fh, indent=2)

    def summary(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for e in self.events:
            counts[e.constraint] = counts.get(e.constraint, 0) + 1
        return counts
