"""Matte job lifecycle as a small state machine."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


class JobStatus(Enum):
    QUEUED = "queued"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"


def _now() -> datetime:
    return datetime.now(timezone.utc)


@dataclass
class MatteJob:
    id: str
    session_id: str
    params: dict[str, Any]
    status: JobStatus = JobStatus.QUEUED
    progress: int = 0
    stage: str | None = None
    result: dict[str, Any] | None = None
    error: str | None = None
    created_at: datetime = field(default_factory=_now)
    started_at: datetime | None = None
    finished_at: datetime | None = None

    @classmethod
    def queued(cls, session_id: str, params: dict[str, Any]) -> MatteJob:
        return cls(id=str(uuid.uuid4()), session_id=session_id, params=params)

    def start(self) -> None:
        self.status = JobStatus.RUNNING
        self.started_at = _now()

    def advance(self, progress: int, stage: str | None = None) -> None:
        self.progress = max(0, min(100, int(progress)))
        if stage is not None:
            self.stage = stage

    def complete(self, result: dict[str, Any]) -> None:
        if self.started_at is None:
            raise ValueError("cannot complete a job that never started")
        self.status = JobStatus.DONE
        self.progress = 100
        self.result = result
        self.finished_at = _now()

    def fail(self, message: str) -> None:
        self.status = JobStatus.FAILED
        self.error = message
        self.finished_at = _now()

    def to_dict(self) -> dict[str, Any]:
        def stamp(value: datetime | None) -> str | None:
            return value.isoformat() if value else None

        return {
            "id": self.id,
            "session_id": self.session_id,
            "status": self.status.value,
            "progress": self.progress,
            "stage": self.stage,
            "params": self.params,
            "result": self.result,
            "error": self.error,
            "created_at": stamp(self.created_at),
            "started_at": stamp(self.started_at),
            "finished_at": stamp(self.finished_at),
        }
