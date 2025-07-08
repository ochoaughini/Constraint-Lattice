# SPDX-License-Identifier: MIT
"""In-memory knowledge store."""

from __future__ import annotations

import hashlib
import pickle
import sqlite3
import zlib
from collections.abc import Callable, Iterable
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np


class MemoryPersistenceAdapter:
    """Persist ``MemoryStore`` contents to a SQLite database."""

    def __init__(self, db_path: str | Path = "memory.db") -> None:
        self.db_path = Path(db_path)
        self._init_db()

    def _init_db(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS memory (
                    key TEXT PRIMARY KEY,
                    embedding BLOB,
                    value BLOB,
                    origin TEXT,
                    lineage TEXT,
                    timestamp TEXT
                )
                """
            )

    def save(self, data: dict[str, dict[str, Any]]) -> None:
        with sqlite3.connect(self.db_path) as conn:
            for key, entry in data.items():
                emb = entry["embedding"].astype(np.float32).tobytes()
                value_blob = zlib.compress(pickle.dumps(entry["value"]))
                meta = entry["metadata"]
                conn.execute(
                    "REPLACE INTO memory (key, embedding, value, origin, lineage, timestamp) VALUES (?, ?, ?, ?, ?, ?)",
                    (
                        key,
                        emb,
                        value_blob,
                        meta.get("origin", ""),
                        meta.get("lineage"),
                        meta.get("timestamp"),
                    ),
                )
            conn.commit()

    def load(self) -> dict[str, dict[str, Any]]:
        data: dict[str, dict[str, Any]] = {}
        if not self.db_path.exists():
            return data
        with sqlite3.connect(self.db_path) as conn:
            for row in conn.execute(
                "SELECT key, embedding, value, origin, lineage, timestamp FROM memory"
            ):
                key = row[0]
                embedding = np.frombuffer(row[1], dtype=np.float32)
                value = pickle.loads(zlib.decompress(row[2]))
                data[key] = {
                    "value": value,
                    "embedding": embedding,
                    "metadata": {
                        "origin": row[3],
                        "lineage": row[4],
                        "timestamp": row[5],
                    },
                }
        return data


class MemoryStore:
    """Tiny key-value store with optional embedding search and persistence."""

    def __init__(
        self,
        embed_fn: Callable[[str], np.ndarray] | None = None,
        *,
        persistence: MemoryPersistenceAdapter | None = None,
    ) -> None:
        self.data: dict[str, dict[str, Any]] = {}
        self.embed_fn = embed_fn or self._default_embed
        self.persistence = persistence
        if self.persistence is not None:
            self.data.update(self.persistence.load())

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
        if self.persistence is not None:
            self.persistence.save({key: self.data[key]})

    def get(self, key: str) -> Any:
        entry = self.data.get(key)
        return entry["value"] if entry else None

    def flush(self) -> None:
        """Persist all stored items to disk."""
        if self.persistence is not None:
            self.persistence.save(self.data)

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
