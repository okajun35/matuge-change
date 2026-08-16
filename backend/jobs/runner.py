"""Runs matting off the request thread and publishes progress through the repository."""

from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any

from backend.jobs.gate import DEFAULT_MAX_WORKERS, matte_slot, max_workers
from backend.jobs.job import MatteJob
from backend.jobs.repository import JobRepository

ProgressReporter = Callable[[int, str], None]
Work = Callable[[ProgressReporter], dict[str, Any]]

__all__ = ["DEFAULT_MAX_WORKERS", "MatteJobRunner", "max_workers"]


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
        # the gate (not the executor's width) is what bounds concurrent mattings, because
        # the synchronous endpoint runs outside this executor and peaks just as high
        with matte_slot():
            try:
                job.complete(work(report))
            except Exception as exc:  # surfaced to the client through the job record
                job.fail(f"{type(exc).__name__}: {exc}")
        self._repository.save(job)
