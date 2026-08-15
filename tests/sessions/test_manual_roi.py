"""手動ROIモード: 顔検出が効かない横顔・目のアップでも抽出を始められる。"""

import numpy as np
import pytest

from backend.lash_extraction import matting as matting_module
from backend.sessions import service as service_module
from backend.sessions.errors import FaceNotDetected
from backend.sessions.service import SessionService
from backend.sessions.store import SessionStore


@pytest.fixture
def profile_service(tmp_path, monkeypatch) -> SessionService:
    """横顔と同じ状況（MediaPipe が顔を検出できない）を再現する。"""
    monkeypatch.setattr(service_module, "detect_landmarks", lambda img: None)
    monkeypatch.setattr(matting_module, "detect_landmarks", lambda img: None)
    return SessionService(SessionStore(str(tmp_path)))


@pytest.fixture
def profile_image() -> np.ndarray:
    return np.random.default_rng(11).integers(0, 256, size=(400, 300, 3), dtype=np.uint8)


class TestCreateWithManualRoi:
    def test_auto_mode_still_requires_a_face(self, profile_service, profile_image):
        with pytest.raises(FaceNotDetected):
            profile_service.create(profile_image, None)

    def test_manual_rect_skips_face_detection(self, profile_service, profile_image):
        result = profile_service.create(profile_image, None, roi_rect=(60, 80, 260, 220))
        sid = result["session_id"]
        assert result["width"] == 200 and result["height"] == 140
        assert profile_service.store.load_meta(sid)["roi"] == [60, 80, 260, 220]
        assert profile_service.store.load_meta(sid)["mode"] == "manual"
        assert "probability" in result["layers"]

    def test_probability_is_not_masked_by_an_eye_prior(self, profile_service, profile_image):
        # 目のランドマークが無いので prior は掛けられない。暗部evidenceをそのまま使う
        sid = profile_service.create(profile_image, None, roi_rect=(60, 80, 260, 220))["session_id"]
        prob = profile_service.store.load_array(sid, "probability")
        assert prob.max() > 0.5

    def test_bare_image_is_aligned_by_the_same_rect(self, profile_service, profile_image):
        sid = profile_service.create(profile_image, profile_image.copy(), roi_rect=(60, 80, 260, 220))[
            "session_id"
        ]
        assert profile_service.store.has_layer(sid, "roi_b")
        assert profile_service.store.load_meta(sid)["has_bare"] is True

    def test_auto_mode_is_recorded_in_meta(self, tmp_path, synthetic_landmarks, monkeypatch):
        monkeypatch.setattr(service_module, "detect_landmarks", lambda img: synthetic_landmarks)
        service = SessionService(SessionStore(str(tmp_path)))
        img = np.random.default_rng(3).integers(0, 256, size=(400, 400, 3), dtype=np.uint8)
        sid = service.create(img, None)["session_id"]
        assert service.store.load_meta(sid)["mode"] == "auto"


class TestRecomposeWithoutLandmarks:
    def test_reports_that_the_session_has_no_landmarks(self, profile_service, profile_image):
        sid = profile_service.create(profile_image, None, roi_rect=(60, 80, 260, 220))["session_id"]
        profile_service.store.save_image(sid, "product_rgba", np.zeros((140, 200, 4), np.uint8))
        with pytest.raises(FaceNotDetected):
            profile_service.recompose(sid, profile_image)

    def test_dest_rect_recomposes_without_landmarks_and_preserves_aspect(
        self, profile_service, profile_image
    ):
        sid = profile_service.create(profile_image, None, roi_rect=(60, 80, 260, 220))["session_id"]
        product = np.zeros((20, 40, 4), np.uint8)
        product[:, :, 2] = 255
        product[:, :, 3] = 255
        profile_service.store.save_image(sid, "product_rgba", product)
        edited = np.zeros((300, 400, 3), np.uint8)

        result = profile_service.recompose(sid, edited, dest_rect=(100, 80, 300, 240))

        assert result["dest_rect"] == [100, 80, 300, 240]
        assert profile_service.store.has_layer(sid, "source_edited")
        out = profile_service.store.load_image(sid, "composite_on_edited")
        assert out.shape == edited.shape
        # 2:1 product fitted into a 200x160 rectangle => 200x100, centered.
        assert np.all(out[110:210, 100:300, 2] > 200)
        assert np.all(out[:110, :, 2] == 0)
        assert np.all(out[210:, :, 2] == 0)
        assert profile_service.store.load_meta(sid)["dest_rect"] == [100, 80, 300, 240]

    def test_dest_rect_is_clipped_and_rejects_small_rect(self, profile_service, profile_image):
        sid = profile_service.create(profile_image, None, roi_rect=(60, 80, 260, 220))["session_id"]
        profile_service.store.save_image(sid, "product_rgba", np.zeros((20, 40, 4), np.uint8))
        edited = np.zeros((300, 400, 3), np.uint8)

        profile_service.recompose(sid, edited, dest_rect=(-10, 80, 300, 400))
        assert profile_service.store.load_meta(sid)["dest_rect"] == [0, 80, 300, 300]

        with pytest.raises(ValueError, match="at least"):
            profile_service.recompose(sid, edited, dest_rect=(1, 1, 10, 30))
