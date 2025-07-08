# SPDX-License-Identifier: MIT
"""In-memory knowledge store."""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Iterable
from datetime import datetime
from typing import Any

import numpy as np


class MemoryStore:
    """Tiny key-value store with optional embedding search."""

    def __init__(self, embed_fn: Callable[[str], np.ndarray] | None = None) -> None:
        self.data: dict[str, dict[str, Any]] = {}
        self.embed_fn = embed_fn or self._default_embed

    def _default_embed(self, text: str) -> np.ndarray:
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        vec = np.frombuffer(digest[:32], dtype=np.uint8).astype(np.float32)
        norm = np.linalg.norm(vec)
        return vec / norm if norm else vec

    def add(
        self, key: str, value: Any, *, origin: str = "", lineage: str | None = None
    ) -> None:
        embedding = self.embed_fn(str(value))
        self.data[key] = {
            "value": value,
            "embedding": embedding,
            "metadata": {
                "timestamp": datetime.utcnow().isoformat(),
                "origin": origin,
                "lineage": lineage,
            },
        }

    def get(self, key: str) -> Any:
        entry = self.data.get(key)
        return entry["value"] if entry else None

    def search(self, text: str) -> Iterable[Any]:
        for k, v in self.data.items():
            if text.lower() in k.lower():
                yield v["value"]

    def search_similar(self, text: str, threshold: float = 0.8) -> Iterable[Any]:
        query = self.embed_fn(text)
        for entry in self.data.values():
            vec = entry["embedding"]
            sim = float(
                np.dot(query, vec) / (np.linalg.norm(query) * np.linalg.norm(vec))
            )
            if sim >= threshold:
                yield entry["value"]
