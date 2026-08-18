"""Provenance: the uploaded sources, the composites and the run history are kept."""

import numpy as np
import pytest

from backend.lash_extraction import matting as matting_module
from backend.sessions import service as service_module
from backend.sessions.service import SessionService
from backend.sessions.store import SessionStore


@pytest.fixture
def service(tmp_path, synthetic_landmarks, monkeypatch) -> SessionService:
    monkeypatch.setattr(service_module, "detect_landmarks", lambda img: synthetic_landmarks)
    monkeypatch.setattr(matting_module, "detect_landmarks", lambda img: synthetic_landmarks)
    return SessionService(SessionStore(str(tmp_path)))


@pytest.fixture
def face_image() -> np.ndarray:
    return np.random.default_rng(3).integers(0, 256, size=(400, 400, 3), dtype=np.uint8)


class TestSourcePersistence:
    def test_create_reports_the_original_eye_area_for_comparison_focus(self, service, face_image):
        result = service.create(face_image, None)

        assert result["source_focus_rect"] == service.store.load_meta(result["session_id"])["roi"]

    def test_worn_image_is_kept(self, service, face_image):
        sid = service.create(face_image, None)["session_id"]
        assert service.store.has_layer(sid, "source_with")
        assert service.store.load_image(sid, "source_with").shape == face_image.shape

    def test_bare_image_is_kept(self, service, face_image):
        sid = service.create(face_image, face_image.copy())["session_id"]
        assert service.store.has_layer(sid, "source_without")

    def test_sources_are_offered_as_layers(self, service, face_image):
        # UI の表示切替で元画像をそのまま確認できる必要がある
        assert "source_with" in service.create(face_image, None)["layers"]
        both = service.create(face_image, face_image.copy())["layers"]
        assert {"source_with", "source_without"} <= set(both)

    def test_edited_image_is_kept(self, service, face_image):
        sid = service.create(face_image, None)["session_id"]
        service.run_matte(sid, None)
        service.recompose(sid, face_image)
        assert service.store.has_layer(sid, "source_edited")
        assert service.store.has_layer(sid, "composite_on_edited")

    def test_automatic_recompose_returns_a_bounded_focus_rect(self, service, face_image):
        sid = service.create(face_image, None)["session_id"]
        service.run_matte(sid, None)

        result = service.recompose(sid, face_image)

        x0, y0, x1, y1 = result["focus_rect"]
        height, width = face_image.shape[:2]
        assert 0 <= x0 < x1 <= width
        assert 0 <= y0 < y1 <= height


class TestRunHistory:
    def test_matting_appends_a_run(self, service, face_image):
        sid = service.create(face_image, None)["session_id"]
        result = service.run_matte(sid, None, fg_thresh=0.6, bg_thresh=0.2)
        runs = service.runs(sid)
        assert len(runs) == 1
        assert runs[0]["params"] == {"fg_thresh": 0.6, "bg_thresh": 0.2, "unknown_band_px": 6}
        assert runs[0]["reconstruction_error"] == result["reconstruction_error"]
        assert runs[0]["created_at"]

    def test_runs_accumulate_in_order(self, service, face_image):
        sid = service.create(face_image, None)["session_id"]
        service.run_matte(sid, None, fg_thresh=0.6)
        service.run_matte(sid, None, fg_thresh=0.8)
        assert [r["params"]["fg_thresh"] for r in service.runs(sid)] == [0.6, 0.8]

    def test_unknown_session_raises(self, service):
        with pytest.raises(LookupError):
            service.runs("nope")

    def test_no_runs_before_matting(self, service, face_image):
        sid = service.create(face_image, None)["session_id"]
        assert service.runs(sid) == []
