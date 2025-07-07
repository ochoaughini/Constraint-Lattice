# SPDX-License-Identifier: MIT
"""In-memory knowledge store."""

from __future__ import annotations

from typing import Dict, Any, Iterable


class MemoryStore:
    """Tiny key-value store with embedding vectors."""

    def __init__(self) -> None:
        self.data: Dict[str, Any] = {}

    def add(self, key: str, value: Any) -> None:
        self.data[key] = value

    def get(self, key: str) -> Any:
        return self.data.get(key)

    def search(self, text: str) -> Iterable[Any]:
        for k, v in self.data.items():
            if text.lower() in k.lower():
                yield v
