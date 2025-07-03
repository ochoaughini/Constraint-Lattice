"""Async runtime helpers.

Provides a singleton background event loop for running coroutines from sync
contexts without blocking the main thread.  This replaces *nest_asyncio* hacks
and avoids Jupyter deadlocks.

Usage::

    from engine.async_utils import run_async

    result = run_async(my_coro())
"""
from __future__ import annotations

import asyncio
import logging
import threading
from types import TracebackType
from typing import Any, Optional, Type, TypeVar

logger = logging.getLogger(__name__)
_T = TypeVar("_T")

# ---------------------------------------------------------------------------
# Background event loop singleton
# ---------------------------------------------------------------------------

_LOOP_THREAD: Optional[threading.Thread] = None
_LOOP: Optional[asyncio.AbstractEventLoop] = None


def _loop_worker(loop: asyncio.AbstractEventLoop):  # pragma: no cover
    """Run *loop* forever until it is closed."""
    asyncio.set_event_loop(loop)
    loop.run_forever()


def _ensure_loop() -> asyncio.AbstractEventLoop:  # pragma: no cover
    global _LOOP, _LOOP_THREAD  # noqa: PLW0603

    if _LOOP and _LOOP.is_running():
        return _LOOP

    # Create a fresh loop detached from any existing event loop (e.g. Jupyter)
    _LOOP = asyncio.new_event_loop()
    _LOOP_THREAD = threading.Thread(target=_loop_worker, args=(_LOOP,), daemon=True)
    _LOOP_THREAD.start()
    logger.info("Background asyncio loop started in thread %s", _LOOP_THREAD.name)
    return _LOOP


# ---------------------------------------------------------------------------
# Public helper
# ---------------------------------------------------------------------------

def run_async(coro: "asyncio.Future[_T] | asyncio.coroutines.Coroutine[Any, Any, _T]") -> _T:  # type: ignore[name-defined]
    """Run *coro* in the background loop and wait for its result synchronously."""
    loop = _ensure_loop()
    fut = asyncio.run_coroutine_threadsafe(coro, loop)
    return fut.result()


# ---------------------------------------------------------------------------
# Context manager for temporary loop usage
# ---------------------------------------------------------------------------

class background_loop:  # noqa: D401 not a public API
    """Context manager that yields the background loop for advanced use cases."""

    def __enter__(self) -> asyncio.AbstractEventLoop:  # noqa: D401
        return _ensure_loop()

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc: Optional[BaseException],
        tb: Optional[TracebackType],
    ) -> None:  # noqa: D401
        # Do not close – keep loop running for subsequent calls
        if exc:
            logger.debug("background_loop exited with %s", exc)
