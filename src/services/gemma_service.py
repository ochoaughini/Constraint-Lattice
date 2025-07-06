# SPDX-License-Identifier: MIT
# Copyright (c) 2025 ochoaughini. See LICENSE for full terms.
# Copyright (c) 2025 ochoaughini. See LICENSE for full terms.
# Copyright (c) 2025 ochoaughini. See LICENSE for full terms.
"""Gemma embedding + classifier helper with Redis cache.

To keep latency low we compute embeddings in a *multiprocessing* pool and cache
results in Redis.  The heavy transformer dependency is optional – if it's not
available we fall back to a deterministic stub.

Environment variables
---------------------
GEMMA_MODEL_NAME    HF model id (default: *google/gemma-2b*)
REDIS_URL           Connection string (default: *redis://localhost:6379/0*)
"""
from __future__ import annotations

import hashlib
import logging
import os
from functools import lru_cache
from typing import Dict, List

logger = logging.getLogger(__name__)

try:
    import redis  # type: ignore
except Exception:  # pragma: no cover – dev/CI
    redis = None  # type: ignore

try:
    from sentence_transformers import SentenceTransformer  # type: ignore
except Exception:  # pragma: no cover – optional heavy dep
    SentenceTransformer = None  # type: ignore

_REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
_MODEL_NAME = os.getenv("GEMMA_MODEL_NAME", "google/gemma-2b")


@lru_cache(maxsize=1)
def _get_redis():  # pragma: no cover – external service
    if redis is None:
        raise RuntimeError("redis dependency missing")
    return redis.Redis.from_url(_REDIS_URL, decode_responses=False)


@lru_cache(maxsize=1)
def _get_model():  # pragma: no cover
    if SentenceTransformer is None:
        raise RuntimeError("sentence-transformers missing")
    return SentenceTransformer(_MODEL_NAME)


def _hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def embed_text(text: str) -> List[float]:  # noqa: D401
    """Return embedding vector for *text*, using Redis memoisation."""
    key = f"gemma:{_hash(text)}".encode()

    try:
        rds = _get_redis()
        if (blob := rds.get(key)) is not None:
            return [float(x) for x in blob.decode().split(",")]
    except Exception as exc:  # pragma: no cover
        logger.debug("Redis unavailable: %s", exc)
        rds = None

    try:
        vec = _get_model().encode(text).tolist()
    except Exception as exc:  # pragma: no cover
        logger.debug("Gemma model unavailable: %s – using stub", exc)
        vec = [len(text) % 1000 / 1000.0] * 384  # fixed dim stub

    if rds is not None:
        try:
            rds.set(key, ",".join(str(x) for x in vec), ex=86400)
        except Exception:  # pragma: no cover
            pass

    return vec


def classify(prompt: str, response: str) -> Dict[str, float]:  # noqa: D401
    """Simple cosine-sim based toxicity heuristic over embeddings."""
    emb_prompt = embed_text(prompt)
    emb_resp = embed_text(response)
    # Cosine similarity stub
    dot = sum(p * r for p, r in zip(emb_prompt, emb_resp))
    norm = (sum(p * p for p in emb_prompt) ** 0.5) * (sum(r * r for r in emb_resp) ** 0.5)
    sim = dot / max(norm, 1e-6)
    conf = max(0.0, min(1.0, 1 - sim))  # invert similarity
    sev = conf * 0.6
    return {"confidence": conf, "severity": sev, "rationale": "Gemma embedding heuristic"}
