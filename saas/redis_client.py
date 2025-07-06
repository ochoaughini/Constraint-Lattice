# SPDX-License-Identifier: BSL-1.1
# Copyright (c) 2025 Lexsight LCC. All rights reserved.
# See saas/LICENSE-BSL.txt for full terms.

This module provides a *singleton* connection pool that can be injected into
FastAPI routes or dependencies.  We deliberately keep the surface small to
avoid leaking implementation details throughout the code base.

Usage in a dependency:

    from fastapi import Depends
    from saas.redis_client import get_redis

    @router.get("/items")
    async def list_items(redis=Depends(get_redis)):
        return await redis.hgetall("items")

The connection URL is taken from the ``REDIS_URL`` environment variable and
falls back to ``redis://localhost:6379/0`` for local Windsurf runs.
"""
from __future__ import annotations

import os
from functools import lru_cache

import redis.asyncio as redis_async
from redis.asyncio.client import Redis

__all__ = ["get_redis", "redis_pool"]


def _redis_url() -> str:
    """Return connection string – coordinates controlled via env var."""
    return os.getenv("REDIS_URL", "redis://localhost:6379/0")


@lru_cache(maxsize=1)
def redis_pool() -> Redis:  # pragma: no cover – side-effectful singleton
    """Create (or return cached) global Redis connection pool."""
    return redis_async.from_url(_redis_url(), encoding="utf-8", decode_responses=True)


async def get_redis() -> Redis:  # noqa: D401 – FastAPI dependency style
    """Yield a Redis client suitable for injection with ``Depends``."""
    pool = redis_pool()
    try:
        yield pool
    finally:
        # Global pool remains open – individual routes should *not* close it.
        pass
