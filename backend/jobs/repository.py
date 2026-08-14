"""Job persistence ports and adapters."""

from __future__ import annotations

from typing import Protocol

from backend.jobs.job import MatteJob


class JobRepository(Protocol):
    def save(self, job: MatteJob) -> None: ...

    def get(self, job_id: str) -> MatteJob | None: ...


class InMemoryJobRepository:
    """Always-on store so job polling works without Supabase."""

    def __init__(self) -> None:
        self._jobs: dict[str, MatteJob] = {}

    def save(self, job: MatteJob) -> None:
        self._jobs[job.id] = job

    def get(self, job_id: str) -> MatteJob | None:
        return self._jobs.get(job_id)


class MirroringJobRepository:
    """Keeps jobs in memory and mirrors every change to Supabase for Realtime."""

    def __init__(self, primary: JobRepository, mirror) -> None:
        self._primary = primary
        self._mirror = mirror

    def save(self, job: MatteJob) -> None:
        self._primary.save(job)
        self._mirror.upsert(job)

    def get(self, job_id: str) -> MatteJob | None:
        return self._primary.get(job_id)
