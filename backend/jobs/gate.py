"""The one place a matting is allowed to run, shared by both API paths."""

from __future__ import annotations

import logging
import os
import threading
from collections.abc import Iterator
from contextlib import contextmanager

from backend.jobs.memory import release_memory

DEFAULT_MAX_WORKERS = 1

logger = logging.getLogger("backend.matte")


def max_workers() -> int:
    """Concurrent mattings, from `MATTE_MAX_WORKERS`. One by default: a single solve is the
    process' memory peak, and two at once is what gets a 512MB host OOM-killed."""
    try:
        value = int(os.environ.get("MATTE_MAX_WORKERS", "").strip())
    except ValueError:
        return DEFAULT_MAX_WORKERS
    return value if value > 0 else DEFAULT_MAX_WORKERS


class _GateState:
    """Semaphore rebuilt whenever the configured limit changes (tests, reconfiguration)."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._limit: int | None = None
        self._semaphore: threading.Semaphore | None = None
        self.active = 0

    def semaphore(self, limit: int) -> threading.Semaphore:
        with self._lock:
            if self._semaphore is None or self._limit != limit:
                self._limit, self._semaphore = limit, threading.Semaphore(limit)
            return self._semaphore

    def enter(self) -> None:
        with self._lock:
            self.active += 1

    def leave(self) -> None:
        with self._lock:
            self.active -= 1


_state = _GateState()


def active_mattings() -> int:
    """How many mattings hold a slot right now."""
    return _state.active


@contextmanager
def matte_slot() -> Iterator[None]:
    """Hold one of the `MATTE_MAX_WORKERS` matting slots, then return the memory to the OS.

    Both `POST /api/matte` (which runs on the request thread) and the queued jobs go
    through here, so the limit bounds the process rather than one code path: without it a
    queued job and a synchronous request peak at the same time and the 512MB host dies
    before either of them reaches its cleanup.
    """
    limit = max_workers()
    semaphore = _state.semaphore(limit)
    if not semaphore.acquire(blocking=False):
        # The wait is the usual explanation for a matting that looks hung with the default
        # limit of one, so it is logged rather than left invisible.
        logger.info("matte waiting for a matting slot: active=%d max_workers=%d", _state.active, limit)
        semaphore.acquire()
    _state.enter()
    try:
        yield
    finally:
        _state.leave()
        semaphore.release()
        release_memory()
