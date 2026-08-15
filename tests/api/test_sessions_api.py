import os

import cv2
import numpy as np

from backend.app import DATA_DIR


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


class TestRunHistoryApi:
    def test_empty_before_matting(self, client, session_id):
        assert client.get(f"/api/sessions/{session_id}/runs").json()["runs"] == []

    def test_run_recorded_after_matting(self, client, matted_session):
        runs = client.get(f"/api/sessions/{matted_session}/runs").json()["runs"]
        assert len(runs) == 1
        assert runs[0]["params"]["fg_thresh"] == 0.70
        assert isinstance(runs[0]["reconstruction_error"], float)

    def test_unknown_session_returns_404(self, client):
        assert client.get("/api/sessions/nope/runs").status_code == 404
