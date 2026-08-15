"""Runs matting off the request thread and publishes progress through the repository."""

from __future__ import annotations

import os
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any

from backend.jobs.job import MatteJob
from backend.jobs.memory import release_memory
from backend.jobs.repository import JobRepository

ProgressReporter = Callable[[int, str], None]
Work = Callable[[ProgressReporter], dict[str, Any]]

DEFAULT_MAX_WORKERS = 1


def max_workers() -> int:
    """Concurrent mattings. One by default: each solve is the process' memory peak, and two
    of them at once is what gets a 512MB host OOM-killed (the second click returning 502)."""
    try:
        value = int(os.environ.get("MATTE_MAX_WORKERS", "").strip())
    except ValueError:
        return DEFAULT_MAX_WORKERS
    return value if value > 0 else DEFAULT_MAX_WORKERS


class MatteJobRunner:
    def __init__(self, repository: JobRepository, workers: int | None = None) -> None:
        self._repository = repository
        self._executor = ThreadPoolExecutor(max_workers=workers or max_workers(), thread_name_prefix="matte")
        self._futures: dict[str, Future] = {}

    def submit(self, session_id: str, params: dict[str, Any], work: Work) -> str:
        job = MatteJob.queued(session_id, params)
        self._repository.save(job)
        self._futures[job.id] = self._executor.submit(self._run, job, work)
        return job.id

    def get(self, job_id: str) -> MatteJob | None:
        return self._repository.get(job_id)

    def wait(self, job_id: str, timeout: float | None = None) -> None:
        future = self._futures.get(job_id)
        if future is not None:
            future.result(timeout=timeout)

    def _run(self, job: MatteJob, work: Work) -> None:
        def report(progress: int, stage: str) -> None:
            job.advance(progress, stage)
            self._repository.save(job)

        job.start()
        self._repository.save(job)
        try:
            job.complete(work(report))
        except Exception as exc:  # surfaced to the client through the job record
            job.fail(f"{type(exc).__name__}: {exc}")
        self._repository.save(job)
        release_memory()
