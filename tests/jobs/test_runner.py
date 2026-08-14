from backend.jobs.job import JobStatus
from backend.jobs.repository import InMemoryJobRepository
from backend.jobs.runner import MatteJobRunner


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
