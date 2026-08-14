import numpy as np
import pytest

from backend import pipeline


class TestBuildTrimap:
    def test_thresholds(self):
        prob = np.zeros((20, 20), np.float64)
        prob[0:5, 0:5] = 0.9  # FG
        prob[10:12, 10:12] = 0.3  # maybe -> unknown
        trimap = pipeline.build_trimap(prob, None, fg_thresh=0.7, bg_thresh=0.18, unknown_band_px=1)
        assert trimap.dtype == np.uint8
        assert (trimap[0:5, 0:5] == 255).all()
        assert (trimap[10:12, 10:12] == 128).all()
        assert trimap[19, 19] == 0

    def test_constraints_override(self):
        prob = np.full((20, 20), 0.9, np.float64)  # all FG without constraints
        constraints = np.zeros((20, 20), np.int8)
        constraints[0:5, :] = -1
        constraints[5:10, :] = 2
        constraints[10:15, :] = 1
        trimap = pipeline.build_trimap(prob, constraints)
        assert (trimap[0:5] == 0).all()
        assert (trimap[5:10] == 128).all()
        assert (trimap[10:15] == 255).all()
        assert (trimap[15:20] == 255).all()  # unconstrained keeps auto FG


class TestInitialProbability:
    def test_multiply_and_clip(self):
        evidence = np.array([[0.5, 2.0], [0.0, 1.0]])
        prior = np.array([[1.0, 1.0], [1.0, 0.5]])
        prob = pipeline.initial_probability(evidence, prior)
        assert prob[0, 0] == pytest.approx(0.5)
        assert prob[0, 1] == pytest.approx(1.0)  # clipped
        assert prob[1, 0] == pytest.approx(0.0)
        assert prob[1, 1] == pytest.approx(0.5)


class TestComposite:
    def test_alpha_blend(self):
        alpha = np.array([[0.0, 1.0, 0.5]])
        fg = np.ones((1, 3, 3), np.float64)  # white foreground
        base = np.zeros((1, 3, 3), np.uint8)  # black base
        out = pipeline.composite(alpha, fg, base)
        assert (out[0, 0] == 0).all()
        assert (out[0, 1] == 255).all()
        assert (np.abs(out[0, 2].astype(int) - 127) <= 1).all()


class TestReconstructionError:
    def test_zero_when_alpha_empty(self):
        alpha = np.zeros((8, 8))
        fg = np.zeros((8, 8, 3))
        roi_a = np.zeros((8, 8, 3), np.uint8)
        assert pipeline.reconstruction_error(alpha, fg, roi_a) == 0.0

    def test_zero_for_perfect_reconstruction(self):
        rng = np.random.default_rng(0)
        roi_a = rng.integers(0, 256, size=(8, 8, 3), dtype=np.uint8)
        alpha = np.ones((8, 8))
        fg = roi_a.astype(np.float64) / 255.0
        assert pipeline.reconstruction_error(alpha, fg, roi_a) == pytest.approx(0.0, abs=1.0)


class TestEyeRoi:
    def test_compute_and_crop(self, synthetic_landmarks):
        img = np.zeros((400, 400, 3), np.uint8)
        roi = pipeline.compute_eye_roi(synthetic_landmarks, img.shape)
        assert 0 <= roi.x0 < roi.x1 <= 400
        assert 0 <= roi.y0 < roi.y1 <= 400
        assert 0 < roi.scale <= 1.0
        crop = pipeline.crop_roi(img, roi)
        assert crop.shape[0] > 0 and crop.shape[1] > 0
        assert crop.shape[1] <= pipeline.MAX_ROI_WIDTH

    def test_downscales_wide_roi(self, synthetic_landmarks):
        lms = synthetic_landmarks * 10  # 4000px-wide face
        img = np.zeros((4000, 4000, 3), np.uint8)
        roi = pipeline.compute_eye_roi(lms, img.shape)
        assert roi.scale < 1.0
        crop = pipeline.crop_roi(img, roi)
        assert crop.shape[1] <= pipeline.MAX_ROI_WIDTH


class TestDifferenceMap:
    def test_identical_images_give_zero(self):
        rng = np.random.default_rng(1)
        img = rng.integers(0, 256, size=(32, 32, 3), dtype=np.uint8)
        d = pipeline.difference_map(img, img)
        assert d.shape == (32, 32)
        assert d.max() == pytest.approx(0.0)

    def test_range(self):
        rng = np.random.default_rng(2)
        a = rng.integers(0, 256, size=(32, 32, 3), dtype=np.uint8)
        b = rng.integers(0, 256, size=(32, 32, 3), dtype=np.uint8)
        d = pipeline.difference_map(a, b)
        assert d.min() >= 0.0 and d.max() <= 1.0


class TestDarknessMap:
    def test_range_and_shape(self):
        rng = np.random.default_rng(3)
        img = rng.integers(0, 256, size=(32, 32, 3), dtype=np.uint8)
        d = pipeline.darkness_map(img)
        assert d.shape == (32, 32)
        assert d.min() >= 0.0 and d.max() <= 1.0


class TestAlignBToA:
    def test_translation_recovered(self, synthetic_landmarks):
        rng = np.random.default_rng(4)
        img_a = rng.integers(0, 256, size=(400, 400, 3), dtype=np.uint8)
        shift = np.array([7.0, -5.0])
        matrix = np.array([[1, 0, shift[0]], [0, 1, shift[1]]], dtype=np.float64)
        import cv2

        img_b = cv2.warpAffine(img_a, matrix, (400, 400), borderMode=cv2.BORDER_REPLICATE)
        lms_b = synthetic_landmarks + shift
        warped = pipeline.align_b_to_a(img_a, synthetic_landmarks, img_b, lms_b)
        center = (slice(100, 300), slice(100, 300))
        err = np.abs(warped[center].astype(int) - img_a[center].astype(int)).mean()
        assert err < 5.0


class TestRecomposeOnto:
    def test_identity_mapping(self, synthetic_landmarks, monkeypatch):
        monkeypatch.setattr(pipeline, "detect_landmarks", lambda _img: synthetic_landmarks)
        roi = pipeline.EyeRoi(0, 0, 400, 400, 1.0)
        rgba = np.zeros((400, 400, 4), np.uint8)
        rgba[100:120, 100:120] = (0, 0, 255, 255)  # opaque red square (BGR)
        edited = np.full((400, 400, 3), 40, np.uint8)
        out = pipeline.recompose_onto(rgba, roi, synthetic_landmarks, edited)
        assert out is not None
        assert (out[105:115, 105:115] == (0, 0, 255)).all()
        assert (out[300:320, 300:320] == 40).all()

    def test_roi_offset_and_scale(self, synthetic_landmarks, monkeypatch):
        monkeypatch.setattr(pipeline, "detect_landmarks", lambda _img: synthetic_landmarks)
        # ROI crop at (50, 60) downscaled by 0.5: ROI pixel (25, 20) -> image (100, 100)
        roi = pipeline.EyeRoi(50, 60, 250, 260, 0.5)
        rgba = np.zeros((100, 100, 4), np.uint8)
        rgba[15:25, 20:30] = (0, 0, 255, 255)
        edited = np.zeros((400, 400, 3), np.uint8)
        out = pipeline.recompose_onto(rgba, roi, synthetic_landmarks, edited)
        assert out is not None
        assert (out[95:105, 95:105, 2] > 200).all()  # red around (100, 100)
        assert (out[300:320, 300:320] == 0).all()

    def test_none_when_no_face(self, synthetic_landmarks, monkeypatch):
        monkeypatch.setattr(pipeline, "detect_landmarks", lambda _img: None)
        roi = pipeline.EyeRoi(0, 0, 100, 100, 1.0)
        rgba = np.zeros((100, 100, 4), np.uint8)
        edited = np.zeros((200, 200, 3), np.uint8)
        assert pipeline.recompose_onto(rgba, roi, synthetic_landmarks, edited) is None


class TestRunMatting:
    def test_alpha_respects_trimap(self):
        rng = np.random.default_rng(5)
        img = rng.integers(0, 256, size=(32, 32, 3), dtype=np.uint8)
        trimap = np.zeros((32, 32), np.uint8)
        trimap[8:24, 8:24] = 128
        trimap[12:20, 12:20] = 255
        alpha, fg = pipeline.run_matting(img, trimap)
        assert alpha.shape == (32, 32)
        assert fg.shape == (32, 32, 3)
        assert (alpha[trimap == 255] > 0.99).all()
        assert (alpha[trimap == 0] < 0.01).all()
        assert alpha.min() >= 0.0 and alpha.max() <= 1.0
