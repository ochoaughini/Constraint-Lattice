"""Asynchronous streaming constraint evaluation with back-pressure.

`apply_stream` consumes an async iterator of (prompt, partial_output) tuples
and yields moderated chunks downstream.  It uses an `asyncio.Queue` to provide
flow-control – downstream can apply back-pressure by awaiting `queue.get()` at
its own pace.

The `maxsize` is sized heuristically to the worst-case LLM token rate
(≈80 tokens/s) * average processing latency (0.4 s) ≈ 32 items.
Override via the `CLATTICE_STREAM_QUEUE` env-var.
"""
from __future__ import annotations

import asyncio
import os
from typing import AsyncIterator, Tuple

from engine.apply import apply_constraints
from engine.policy_loader import load_constraints

logger = from constraint_lattice.logging_config import configure_logger
logger = configure_logger(__name__)(__name__)

_DEFAULT_QSIZE = int(os.getenv("CLATTICE_STREAM_QUEUE", "32"))


async def apply_stream(
    chunks: AsyncIterator[Tuple[str, str]],
    *,
    constraints=None,
    qsize: int = _DEFAULT_QSIZE,
):
    """Yield moderated outputs preserving order while respecting back-pressure."""
    if constraints is None:
        constraints = load_constraints()

    queue: asyncio.Queue[str] = asyncio.Queue(maxsize=qsize)

    async def _producer():
        async for prompt, output in chunks:
            moderated = apply_constraints(prompt, output, constraints)
            await queue.put(moderated)
        await queue.put(None)  # sentinel

    async def _consumer():
        while True:
            item = await queue.get()
            if item is None:
                break
            yield item

    # Run producer task and consume inline
    prod_task = asyncio.create_task(_producer())
    async for result in _consumer():
        yield result
    await prod_task
