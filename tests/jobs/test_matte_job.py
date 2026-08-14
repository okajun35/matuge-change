import pytest

from backend.jobs.job import JobStatus, MatteJob
from backend.jobs.repository import InMemoryJobRepository


@pytest.fixture
def job() -> MatteJob:
    return MatteJob.queued("sess1", {"fg_thresh": 0.7})


class TestMatteJob:
    def test_new_job_is_queued(self, job):
        assert job.status is JobStatus.QUEUED
        assert job.progress == 0
        assert job.started_at is None

    def test_start_marks_running_with_timestamp(self, job):
        job.start()
        assert job.status is JobStatus.RUNNING
        assert job.started_at is not None

    def test_advance_updates_progress_and_stage(self, job):
        job.start()
        job.advance(40, "alpha")
        assert job.progress == 40
        assert job.stage == "alpha"

    def test_progress_is_clamped(self, job):
        job.start()
        job.advance(140, "alpha")
        assert job.progress == 100
        job.advance(-5, "alpha")
        assert job.progress == 0

    def test_complete_stores_result_and_full_progress(self, job):
        job.start()
        job.complete({"layers": ["alpha"], "reconstruction_error": 1.5})
        assert job.status is JobStatus.DONE
        assert job.progress == 100
        assert job.result["layers"] == ["alpha"]
        assert job.finished_at is not None

    def test_fail_stores_error(self, job):
        job.start()
        job.fail("boom")
        assert job.status is JobStatus.FAILED
        assert job.error == "boom"
        assert job.finished_at is not None

    def test_cannot_complete_a_job_that_never_started(self, job):
        with pytest.raises(ValueError):
            job.complete({})

    def test_to_dict_is_json_friendly(self, job):
        job.start()
        job.complete({"layers": []})
        payload = job.to_dict()
        assert payload["status"] == "done"
        assert payload["session_id"] == "sess1"
        assert isinstance(payload["created_at"], str)


class TestInMemoryJobRepository:
    def test_saved_job_can_be_fetched(self, job):
        repo = InMemoryJobRepository()
        repo.save(job)
        assert repo.get(job.id) is job

    def test_missing_job_returns_none(self):
        assert InMemoryJobRepository().get("nope") is None
