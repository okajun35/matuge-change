"""解析開始（POST /api/session）のメモリ上限対策。

スマホ実写（12MP など）でも 512MB ホストで落ちないように、
- 顔検出は縮小コピーで行う（MediaPipe に巨大画像を渡さない）
- 元画像はデコードを1枚ずつにして、片方を解放してから次を読む
- 位置合わせは全画面ではなく ROI 窓に直接 warp する
という不変条件をテストで固定する。
"""

import gc
import os
import weakref

import cv2
import numpy as np
import pytest

from backend.lash_extraction import alignment
from backend.lash_extraction import landmarks as landmarks_module
from backend.lash_extraction import roi as roi_module
from backend.lash_extraction.roi import EyeRoi
from backend.sessions import service as service_module
from backend.sessions.errors import FaceNotDetected
from backend.sessions.service import SessionService
from backend.sessions.store import SessionStore


class TestDetectLandmarksDownscales:
    """MediaPipe は内部で入力を縮小するので、巨大画像をそのまま渡す意味がない。"""

    def test_large_image_is_downscaled_before_detection(self, monkeypatch):
        seen: dict[str, tuple[int, int]] = {}

        class FakeLandmark:
            def __init__(self, x, y):
                self.x, self.y = x, y

        class FakeResult:
            face_landmarks = [[FakeLandmark(0.25, 0.5)]]

        class FakeLandmarker:
            def detect(self, image):
                seen["shape"] = image.numpy_view().shape[:2]
                return FakeResult()

        monkeypatch.setattr(landmarks_module, "get_landmarker", lambda: FakeLandmarker())
        big = np.zeros((3000, 4000, 3), np.uint8)
        lms = landmarks_module.detect_landmarks(big)

        h, w = seen["shape"]
        assert max(h, w) <= landmarks_module.DETECT_MAX_SIDE
        assert (w / h) == pytest.approx(4000 / 3000, abs=1e-2)
        # 座標は元画像のピクセル系で返す
        assert lms[0] == pytest.approx([0.25 * 4000, 0.5 * 3000], abs=1.0)

    def test_small_image_is_passed_through(self, monkeypatch):
        seen: dict[str, tuple[int, int]] = {}

        class FakeLandmarker:
            def detect(self, image):
                seen["shape"] = image.numpy_view().shape[:2]
                return type("R", (), {"face_landmarks": []})()

        monkeypatch.setattr(landmarks_module, "get_landmarker", lambda: FakeLandmarker())
        small = np.zeros((400, 400, 3), np.uint8)
        assert landmarks_module.detect_landmarks(small) is None
        assert seen["shape"] == (400, 400)


class TestAlignIntoWindow:
    """全画面 warp（元画像と同サイズの一時画像）を作らずに ROI 窓だけを得る。"""

    def test_matches_full_warp_then_crop(self, synthetic_landmarks):
        rng = np.random.default_rng(11)
        img_a = rng.integers(0, 256, size=(400, 400, 3), dtype=np.uint8)
        matrix = np.array([[1, 0, 6.0], [0, 1, -4.0]], dtype=np.float64)
        img_b = cv2.warpAffine(img_a, matrix, (400, 400), borderMode=cv2.BORDER_REPLICATE)
        lms_b = synthetic_landmarks + np.array([6.0, -4.0])
        roi = EyeRoi(37, 51, 337, 291, 1.0)

        full = roi_module.crop_roi(alignment.align_b_to_a(img_a, synthetic_landmarks, img_b, lms_b), roi)
        windowed = alignment.align_b_into_roi(img_b, lms_b, synthetic_landmarks, roi)
        assert windowed.shape == full.shape
        assert np.array_equal(windowed, full)

    def test_matches_full_warp_then_crop_when_downscaled(self, synthetic_landmarks):
        rng = np.random.default_rng(12)
        img_a = rng.integers(0, 256, size=(400, 400, 3), dtype=np.uint8)
        matrix = np.array([[1, 0, 3.0], [0, 1, 2.0]], dtype=np.float64)
        img_b = cv2.warpAffine(img_a, matrix, (400, 400), borderMode=cv2.BORDER_REPLICATE)
        lms_b = synthetic_landmarks + np.array([3.0, 2.0])
        roi = EyeRoi(20, 30, 380, 330, 0.5)

        full = roi_module.crop_roi(alignment.align_b_to_a(img_a, synthetic_landmarks, img_b, lms_b), roi)
        windowed = alignment.align_b_into_roi(img_b, lms_b, synthetic_landmarks, roi)
        assert np.array_equal(windowed, full)


class TestCropOwnsItsMemory:
    def test_crop_does_not_pin_the_source_image(self):
        img = np.zeros((400, 400, 3), np.uint8)
        crop = roi_module.crop_roi(img, EyeRoi(10, 10, 110, 110, 1.0))
        assert crop.base is None, "a view would keep the whole source image alive"


@pytest.fixture
def service(tmp_path, synthetic_landmarks, monkeypatch) -> SessionService:
    monkeypatch.setattr(service_module, "detect_landmarks", lambda img: synthetic_landmarks)
    return SessionService(SessionStore(str(tmp_path)))


@pytest.fixture
def face_image() -> np.ndarray:
    return np.random.default_rng(5).integers(0, 256, size=(400, 400, 3), dtype=np.uint8)


class TestCreateLogsTheSourceSize:
    def test_worn_image_size_is_logged_before_detection(self, service, face_image, caplog):
        # 落ちたときに「どのサイズの写真で落ちたか」がログだけで分かるようにする
        with caplog.at_level("INFO", logger="backend.matte"):
            service.create(face_image, None)
        assert "worn image 400x400" in caplog.text


class TestLazyDecode:
    """2枚同時にフル解像度で抱えない（12MP×2 で 512MB ホストが落ちる原因）。"""

    def test_bare_image_is_decoded_after_the_worn_one_is_released(self, service, face_image):
        ref: dict[str, weakref.ref] = {}

        def load_a():
            img = face_image.copy()
            ref["a"] = weakref.ref(img)
            return img

        def load_b():
            gc.collect()
            assert ref["a"]() is None, "worn image must be released before decoding the bare one"
            return face_image.copy()

        body = service.create_lazily(load_a, load_b)
        assert body["has_bare"] is True
        assert service.store.has_layer(body["session_id"], "source_without")

    def test_matches_the_eager_api(self, service, face_image):
        eager = service.create(face_image, face_image.copy())
        lazy = service.create_lazily(lambda: face_image.copy(), lambda: face_image.copy())
        for key in ("width", "height", "has_bare", "mode", "layers"):
            assert lazy[key] == eager[key]
        assert np.array_equal(
            service.store.load_array(eager["session_id"], "probability"),
            service.store.load_array(lazy["session_id"], "probability"),
        )

    def test_no_session_is_left_behind_when_the_bare_image_has_no_face(
        self, service, face_image, monkeypatch, synthetic_landmarks
    ):
        calls: list[int] = []

        def detect(img):
            calls.append(1)
            return synthetic_landmarks if len(calls) == 1 else None

        monkeypatch.setattr(service_module, "detect_landmarks", detect)
        before = set(os.listdir(service.store.root))
        with pytest.raises(FaceNotDetected):
            service.create_lazily(lambda: face_image, lambda: face_image.copy())
        assert set(os.listdir(service.store.root)) == before
