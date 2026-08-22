"""Asynchronous, pollable jobs for long-running video analysis and compositing."""

from __future__ import annotations

import threading
import uuid
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any

VideoProgressReporter = Callable[[str, int, int | None, int | None, str], None]
VideoWork = Callable[[VideoProgressReporter], dict[str, Any]]


@dataclass
class VideoJob:
    """Mutable job state, guarded by its own lock for HTTP polling."""

    id: str
    session_id: str
    status: str = "queued"
    phase: str = "queued"
    progress: int = 0
    processed_frames: int | None = None
    total_frames: int | None = None
    message: str = "処理を待っています"
    result: dict[str, Any] | None = None
    error: str | None = None
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def report(
        self,
        phase: str,
        progress: int,
        processed_frames: int | None,
        total_frames: int | None,
        message: str,
    ) -> None:
        with self._lock:
            self.status = "running"
            self.phase = phase
            self.progress = max(0, min(100, progress))
            self.processed_frames = processed_frames
            self.total_frames = total_frames
            self.message = message

    def complete(self, result: dict[str, Any]) -> None:
        with self._lock:
            self.status = "done"
            self.phase = "done"
            self.progress = 100
            self.message = "処理が完了しました"
            self.result = result

    def fail(self, error: Exception) -> None:
        with self._lock:
            self.status = "failed"
            self.phase = "failed"
            self.message = "処理を完了できませんでした"
            self.error = f"{type(error).__name__}: {error}"

    def to_dict(self) -> dict[str, Any]:
        with self._lock:
            return {
                "id": self.id,
                "session_id": self.session_id,
                "status": self.status,
                "phase": self.phase,
                "progress": self.progress,
                "processed_frames": self.processed_frames,
                "total_frames": self.total_frames,
                "message": self.message,
                "result": self.result,
                "error": self.error,
            }


class VideoJobRunner:
    """Runs one video at a time to keep memory use bounded and exposes status for polling."""

    def __init__(self, workers: int = 1) -> None:
        self._executor = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="video")
        self._jobs: dict[str, VideoJob] = {}
        self._futures: dict[str, Future] = {}
        self._lock = threading.Lock()

    def submit(self, session_id: str, work: VideoWork) -> str:
        job = VideoJob(id=uuid.uuid4().hex, session_id=session_id)
        with self._lock:
            self._jobs[job.id] = job
            self._futures[job.id] = self._executor.submit(self._run, job, work)
        return job.id

    def get(self, job_id: str) -> VideoJob | None:
        with self._lock:
            return self._jobs.get(job_id)

    def wait(self, job_id: str, timeout: float | None = None) -> None:
        with self._lock:
            future = self._futures.get(job_id)
        if future is not None:
            future.result(timeout=timeout)

    @staticmethod
    def _run(job: VideoJob, work: VideoWork) -> None:
        job.report("starting", 0, None, None, "処理を開始しています")
        try:
            job.complete(work(job.report))
        except Exception as exc:
            job.fail(exc)
