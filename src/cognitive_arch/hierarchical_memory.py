from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

class HierarchicalMemory:
    """Simple persistent hierarchical memory.

    Stores nested symbolic data structures to simulate a distributed memory.
    Data is persisted to a JSON file so it can be reloaded across sessions.
    This is a lightweight stand-in for the advanced memory described in the
    research notes.
    """

    def __init__(self, path: str | Path = "memory.json") -> None:
        self.path = Path(path)
        if self.path.exists():
            with self.path.open("r", encoding="utf-8") as fh:
                self._data: Dict[str, Any] = json.load(fh)
        else:
            self._data = {}

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("w", encoding="utf-8") as fh:
            json.dump(self._data, fh, indent=2)

    def add(self, keys: List[str], value: Any) -> None:
        """Add a value under a hierarchy of keys."""
        node = self._data
        for key in keys[:-1]:
            node = node.setdefault(key, {})
        node[keys[-1]] = value
        self._save()

    def get(self, keys: List[str]) -> Any:
        node = self._data
        for key in keys:
            node = node.get(key, {})
        return node

    def search(self, key: str) -> List[Any]:
        """Return all values associated with key anywhere in the hierarchy."""
        results = []
        stack = [self._data]
        while stack:
            current = stack.pop()
            if isinstance(current, dict):
                for k, v in current.items():
                    if k == key:
                        results.append(v)
                    if isinstance(v, dict):
                        stack.append(v)
        return results
