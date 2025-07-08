from __future__ import annotations

import json
from typing import Iterable, List


class ConstraintCompiler:
    """Generate constraint definitions from text or JSON."""

    @staticmethod
    def from_text(lines: Iterable[str]) -> List[str]:
        return [line.strip() for line in lines if line.strip()]

    @staticmethod
    def from_json(data: str) -> List[str]:
        obj = json.loads(data)
        if isinstance(obj, list):
            return [str(x) for x in obj]
        return [f"{k}: {v}" for k, v in obj.items()]

