import json
import os
import shutil
import threading

import cv2
import numpy as np

from backend.api.container import container
from backend.app import DATA_DIR
from backend.jobs import gate as gate_module


class TestGetSession:
    def test_returns_available_layers(self, client, session_id):
        body = client.get(f"/api/sessions/{session_id}").json()
        assert body["width"] == 48 and body["height"] == 48
        assert body["layers"] == ["roi_a"]

    def test_layers_grow_after_matting(self, client, matted_session):
        layers = client.get(f"/api/sessions/{matted_session}").json()["layers"]
        assert layers == ["roi_a", "trimap", "alpha", "product_rgba"]

    def test_source_layers_are_listed_next_to_the_roi(self, client, session_id):
        # UI の表示プルダウンで元画像を ROI の直後に選べるようにする
        sdir = os.path.join(DATA_DIR, session_id)
        for name in ("source_with", "source_without", "source_edited"):
            cv2.imwrite(os.path.join(sdir, f"{name}.png"), np.zeros((10, 10, 3), np.uint8))
        layers = client.get(f"/api/sessions/{session_id}").json()["layers"]
        assert layers == ["roi_a", "source_with", "source_without", "source_edited"]

    def test_unknown_session_returns_404(self, client):
        assert client.get("/api/sessions/nope").status_code == 404

    def test_returns_persisted_dest_rect(self, client, session_id):
        meta_path = os.path.join(DATA_DIR, session_id, "meta.json")
        with open(meta_path) as f:
            meta = json.load(f)
        meta["dest_rect"] = [1, 2, 30, 40]
        with open(meta_path, "w") as f:
            json.dump(meta, f)
        assert client.get(f"/api/sessions/{session_id}").json()["dest_rect"] == [1, 2, 30, 40]


class TestSessionArchiveApi:
    """Supabase Storage への退避／復元。未設定ならローカル動作のまま 503 を返す。"""

    def test_export_is_unavailable_without_supabase(self, client, session_id):
        response = client.post(f"/api/sessions/{session_id}/archive")
        assert response.status_code == 503
        assert "supabase" in response.json()["detail"].lower()

    def test_restore_is_unavailable_without_supabase(self, client):
        assert client.post("/api/sessions/whatever/archive/restore").status_code == 503

    def test_export_round_trips_through_the_archive(self, client, session_id, fake_archive):
        exported = client.post(f"/api/sessions/{session_id}/archive")
        assert exported.status_code == 200
        assert "roi_a.png" in exported.json()["files"]

        shutil.rmtree(os.path.join(DATA_DIR, session_id))
        assert client.get(f"/api/sessions/{session_id}").status_code == 404

        restored = client.post(f"/api/sessions/{session_id}/archive/restore")
        assert restored.status_code == 200
        assert client.get(f"/api/sessions/{session_id}").json()["layers"] == ["roi_a"]

    def test_export_of_unknown_session_returns_404(self, client, fake_archive):
        assert client.post("/api/sessions/nope/archive").status_code == 404

    def test_restore_of_unknown_session_returns_404(self, client, fake_archive):
        assert client.post("/api/sessions/nope/archive/restore").status_code == 404


class TestRunHistoryApi:
    def test_empty_before_matting(self, client, session_id):
        assert client.get(f"/api/sessions/{session_id}/runs").json()["runs"] == []

    def test_run_recorded_after_matting(self, client, matted_session):
        runs = client.get(f"/api/sessions/{matted_session}/runs").json()["runs"]
        assert len(runs) == 1
        assert runs[0]["params"]["fg_thresh"] == 0.70
        assert isinstance(runs[0]["reconstruction_error"], float)

    def test_run_records_which_solve_produced_it(self, client, matted_session):
        """Tiled results are approximations, so a stored layer must say how it was solved."""
        runs = client.get(f"/api/sessions/{matted_session}/runs").json()["runs"]
        assert runs[0]["solve_mode"] == "full"
        assert runs[0]["max_solve_pixels"] is None

    def test_a_tiled_run_records_its_budget(self, client, session_id, monkeypatch):
        monkeypatch.setenv("MATTE_SOLVE_MODE", "tiled")
        monkeypatch.setenv("MATTE_MAX_SOLVE_PIXELS", "20000")

        assert client.post("/api/matte", data={"session_id": session_id}).status_code == 200

        runs = client.get(f"/api/sessions/{session_id}/runs").json()["runs"]
        assert runs[-1]["solve_mode"] == "tiled"
        assert runs[-1]["max_solve_pixels"] == 20000

    def test_unknown_session_returns_404(self, client):
        assert client.get("/api/sessions/nope/runs").status_code == 404


class TestMattingIsSerialisedAcrossBothApis:
    """`MATTE_MAX_WORKERS=1` must bound the sync endpoint and the job queue together."""

    def test_the_sync_endpoint_waits_for_a_running_job(self, client, session_id, monkeypatch):
        monkeypatch.delenv("MATTE_MAX_WORKERS", raising=False)
        observed: list[int] = []
        job_started, let_job_finish = threading.Event(), threading.Event()

        def slow_matte(*_args, **_kwargs):
            observed.append(gate_module.active_mattings())
            job_started.set()
            let_job_finish.wait(timeout=10)
            return {"layers": []}

        monkeypatch.setattr(container().sessions, "run_matte", slow_matte)
        job = client.post("/api/matte/jobs", data={"session_id": session_id}).json()["job_id"]
        assert job_started.wait(timeout=10)

        sync_done = threading.Event()

        def call_sync():
            client.post("/api/matte", data={"session_id": session_id})
            sync_done.set()

        caller = threading.Thread(target=call_sync)
        caller.start()
        assert not sync_done.wait(timeout=1.0)  # blocked behind the job's slot

        let_job_finish.set()
        caller.join(timeout=10)
        container().jobs.wait(job, timeout=10)

        assert observed == [1, 1]  # never two mattings inside the gate at once


class TestSynchronousMatteReleasesMemory:
    """The sync endpoint peaks as high as the job runner, so it needs the same boundary."""

    def test_memory_is_released_after_a_successful_matte(self, client, session_id, monkeypatch):
        calls: list[str] = []
        monkeypatch.setattr(gate_module, "release_memory", lambda: calls.append("released"))

        assert client.post("/api/matte", data={"session_id": session_id}).status_code == 200
        assert calls == ["released"]

    def test_memory_is_released_after_a_failed_matte(self, client, session_id, monkeypatch):
        calls: list[str] = []
        monkeypatch.setattr(gate_module, "release_memory", lambda: calls.append("released"))
        monkeypatch.setenv("MATTE_SOLVE_MODE", "turbo")  # rejected configuration

        assert client.post("/api/matte", data={"session_id": session_id}).status_code >= 400
        assert calls == ["released"]
