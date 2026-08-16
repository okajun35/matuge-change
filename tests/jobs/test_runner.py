import threading
import time

from backend.jobs.job import JobStatus
from backend.jobs.repository import InMemoryJobRepository
from backend.jobs.runner import MatteJobRunner, max_workers


class TestMatteJobRunner:
    def test_successful_run_records_progress_and_result(self):
        repo = InMemoryJobRepository()
        runner = MatteJobRunner(repo)
        seen = []

        def work(report):
            report(50, "alpha")
            seen.append("ran")
            return {"layers": ["alpha"]}

        job_id = runner.submit("sess1", {"fg_thresh": 0.7}, work)
        runner.wait(job_id, timeout=10)

        job = repo.get(job_id)
        assert seen == ["ran"]
        assert job.status is JobStatus.DONE
        assert job.progress == 100
        assert job.result == {"layers": ["alpha"]}

    def test_failure_is_captured_on_the_job(self):
        repo = InMemoryJobRepository()
        runner = MatteJobRunner(repo)

        def work(report):
            raise RuntimeError("matting exploded")

        job_id = runner.submit("sess1", {}, work)
        runner.wait(job_id, timeout=10)

        job = repo.get(job_id)
        assert job.status is JobStatus.FAILED
        assert "matting exploded" in job.error

    def test_job_is_registered_before_work_finishes(self):
        repo = InMemoryJobRepository()
        runner = MatteJobRunner(repo)
        job_id = runner.submit("sess1", {}, lambda report: {"layers": []})
        assert repo.get(job_id) is not None
        runner.wait(job_id, timeout=10)

    def test_jobs_do_not_run_concurrently_by_default(self):
        """Two mattings at once would double the peak memory and get the process OOM-killed."""
        runner = MatteJobRunner(InMemoryJobRepository())
        overlapped = []
        active = [0]
        lock = threading.Lock()

        def work(_report):
            with lock:
                active[0] += 1
                overlapped.append(active[0] > 1)
            time.sleep(0.2)
            with lock:
                active[0] -= 1
            return {"layers": []}

        first = runner.submit("sess1", {}, work)
        second = runner.submit("sess1", {}, work)
        runner.wait(first, timeout=10)
        runner.wait(second, timeout=10)

        assert overlapped == [False, False]

    def test_memory_is_released_after_every_job(self, monkeypatch):
        released = []
        monkeypatch.setattr("backend.jobs.gate.release_memory", lambda: released.append(True))
        runner = MatteJobRunner(InMemoryJobRepository())

        runner.wait(runner.submit("sess1", {}, lambda report: {"layers": []}), timeout=10)
        runner.wait(runner.submit("sess1", {}, lambda report: 1 / 0), timeout=10)

        assert len(released) == 2

    def test_worker_count_comes_from_the_environment(self, monkeypatch):
        monkeypatch.delenv("MATTE_MAX_WORKERS", raising=False)
        assert max_workers() == 1
        monkeypatch.setenv("MATTE_MAX_WORKERS", "3")
        assert max_workers() == 3
        monkeypatch.setenv("MATTE_MAX_WORKERS", "nonsense")
        assert max_workers() == 1
