import time


def wait_for(client, job_id: str, timeout: float = 60.0) -> dict:
    deadline = time.time() + timeout
    while time.time() < deadline:
        body = client.get(f"/api/matte/jobs/{job_id}").json()
        if body["status"] in ("done", "failed"):
            return body
        time.sleep(0.1)
    raise AssertionError("job did not finish in time")


class TestMatteJobs:
    def test_job_is_accepted_and_completes(self, client, session_id):
        res = client.post("/api/matte/jobs", data={"session_id": session_id})
        assert res.status_code == 202
        job_id = res.json()["job_id"]
        assert res.json()["status"] == "queued"

        job = wait_for(client, job_id)
        assert job["status"] == "done"
        assert job["progress"] == 100
        assert "product_rgba" in job["result"]["layers"]

    def test_failed_job_reports_the_error(self, client):
        res = client.post("/api/matte/jobs", data={"session_id": "missing-session"})
        assert res.status_code == 202
        job = wait_for(client, res.json()["job_id"])
        assert job["status"] == "failed"
        assert job["error"]

    def test_unknown_job_returns_404(self, client):
        assert client.get("/api/matte/jobs/unknown").status_code == 404


class TestConfig:
    def test_config_exposes_realtime_flag_without_secrets(self, client):
        body = client.get("/api/config").json()["supabase"]
        assert set(body) == {"enabled", "url", "publishable_key", "realtime"}
