import numpy as np
import pytest

from backend import lash_extraction as pipeline
from backend.lash_extraction import matting as matting_module


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


class TestManualEyeRoi:
    """横顔・目のアップは顔検出が効かないため、ユーザーが矩形でROIを与える。"""

    def test_uses_the_given_rect(self):
        img = np.zeros((400, 400, 3), np.uint8)
        roi = pipeline.manual_eye_roi((100, 120, 300, 260), img.shape)
        assert (roi.x0, roi.y0, roi.x1, roi.y1) == (100, 120, 300, 260)
        assert roi.scale == 1.0
        assert pipeline.crop_roi(img, roi).shape[:2] == (140, 200)

    def test_normalises_reversed_and_out_of_bounds_rects(self):
        roi = pipeline.manual_eye_roi((520, 300, 200, -40), (400, 400, 3))
        assert (roi.x0, roi.y0, roi.x1, roi.y1) == (200, 0, 400, 300)

    def test_downscales_wide_rect(self):
        img = np.zeros((4000, 4000, 3), np.uint8)
        roi = pipeline.manual_eye_roi((0, 0, 4000, 2000), img.shape)
        assert roi.scale < 1.0
        assert pipeline.crop_roi(img, roi).shape[1] <= pipeline.MAX_ROI_WIDTH

    def test_rejects_a_degenerate_rect(self):
        with pytest.raises(ValueError):
            pipeline.manual_eye_roi((10, 10, 15, 200), (400, 400, 3))


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
        monkeypatch.setattr(matting_module, "detect_landmarks", lambda _img: synthetic_landmarks)
        roi = pipeline.EyeRoi(0, 0, 400, 400, 1.0)
        rgba = np.zeros((400, 400, 4), np.uint8)
        rgba[100:120, 100:120] = (0, 0, 255, 255)  # opaque red square (BGR)
        edited = np.full((400, 400, 3), 40, np.uint8)
        out = pipeline.recompose_onto(rgba, roi, synthetic_landmarks, edited)
        assert out is not None
        assert (out[105:115, 105:115] == (0, 0, 255)).all()
        assert (out[300:320, 300:320] == 40).all()

    def test_roi_offset_and_scale(self, synthetic_landmarks, monkeypatch):
        monkeypatch.setattr(matting_module, "detect_landmarks", lambda _img: synthetic_landmarks)
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
        monkeypatch.setattr(matting_module, "detect_landmarks", lambda _img: None)
        roi = pipeline.EyeRoi(0, 0, 100, 100, 1.0)
        rgba = np.zeros((100, 100, 4), np.uint8)
        edited = np.zeros((200, 200, 3), np.uint8)
        assert pipeline.recompose_onto(rgba, roi, synthetic_landmarks, edited) is None


class TestBlendRgbaOver:
    """The blend is the biggest single allocation of a recompose, so it runs in float32."""

    def test_alpha_decides_between_product_and_base(self):
        rgba = np.zeros((4, 4, 4), np.uint8)
        rgba[0, 0] = (10, 20, 30, 255)
        rgba[1, 1] = (10, 20, 30, 128)
        base = np.full((4, 4, 3), 200, np.uint8)

        out = matting_module.blend_rgba_over(rgba, base)

        assert out.dtype == np.uint8
        assert (out[0, 0] == (10, 20, 30)).all()
        assert np.abs(out[1, 1].astype(int) - [105, 110, 115]).max() <= 1
        assert (out[3, 3] == 200).all()

    def test_it_is_the_float32_blend_bit_for_bit(self):
        rng = np.random.default_rng(21)
        rgba = rng.integers(0, 256, size=(64, 64, 4), dtype=np.uint8)
        base = rng.integers(0, 256, size=(64, 64, 3), dtype=np.uint8)

        out = matting_module.blend_rgba_over(rgba, base)

        alpha = rgba[..., 3:4].astype(np.float32) / 255.0
        expected = base.astype(np.float32)
        expected += alpha * (rgba[..., :3].astype(np.float32) - expected)
        assert np.array_equal(out, np.clip(expected, 0, 255).astype(np.uint8))

    def test_recompose_onto_blends_through_it(self, synthetic_landmarks, monkeypatch):
        monkeypatch.setattr(matting_module, "detect_landmarks", lambda _img: synthetic_landmarks)
        calls: list[tuple[int, ...]] = []
        original = matting_module.blend_rgba_over

        def spy(rgba, base_bgr):
            calls.append(base_bgr.shape)
            return original(rgba, base_bgr)

        monkeypatch.setattr(matting_module, "blend_rgba_over", spy)
        roi = pipeline.EyeRoi(0, 0, 400, 400, 1.0)
        rgba = np.zeros((400, 400, 4), np.uint8)
        edited = np.full((400, 400, 3), 40, np.uint8)

        assert matting_module.recompose_onto(rgba, roi, synthetic_landmarks, edited) is not None
        assert calls == [(400, 400, 3)]


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


def _strip_trimap(h: int, w: int) -> np.ndarray:
    """Lash-like horizontal strip of FG surrounded by an unknown band."""
    trimap = np.zeros((h, w), np.uint8)
    trimap[h // 2 - 8 : h // 2 + 8, w // 4 : 3 * w // 4] = 128
    trimap[h // 2 - 3 : h // 2 + 3, w // 4 + 4 : 3 * w // 4 - 4] = 255
    return trimap


class TestSolveWindow:
    def test_bounds_the_unknown_and_foreground_with_a_margin(self):
        trimap = np.zeros((100, 200), np.uint8)
        trimap[40:60, 80:120] = 128
        trimap[45:55, 90:110] = 255
        window = matting_module.solve_window(trimap, margin=5)
        assert window == (35, 65, 75, 125)

    def test_margin_is_clipped_to_the_image(self):
        trimap = np.zeros((20, 20), np.uint8)
        trimap[0:3, 17:20] = 128
        assert matting_module.solve_window(trimap, margin=8) == (0, 11, 9, 20)

    def test_none_when_everything_is_known_background(self):
        assert matting_module.solve_window(np.zeros((20, 20), np.uint8)) is None


def _spy_on_solver(monkeypatch) -> list[tuple[int, int]]:
    """Records the shape of every image handed to the closed-form solver."""
    seen: list[tuple[int, int]] = []
    original = matting_module.estimate_alpha_cf

    def spy(image, trimap, *args, **kwargs):
        seen.append(image.shape[:2])
        return original(image, trimap, *args, **kwargs)

    monkeypatch.setattr(matting_module, "estimate_alpha_cf", spy)
    return seen


class TestMattingMemoryBudget:
    def test_solver_only_sees_the_window_around_the_product(self, monkeypatch):
        seen = _spy_on_solver(monkeypatch)
        rng = np.random.default_rng(3)
        img = rng.integers(0, 256, size=(240, 320, 3), dtype=np.uint8)
        trimap = _strip_trimap(240, 320)

        alpha, fg = matting_module.run_matting(img, trimap)

        solved_h, solved_w = seen[0]
        assert solved_h * solved_w < 240 * 320 // 2
        assert alpha.shape == (240, 320)
        assert fg.shape == (240, 320, 3)

    def test_outside_the_window_alpha_is_background_and_fg_is_the_source(self):
        rng = np.random.default_rng(4)
        img = rng.integers(0, 256, size=(200, 160, 3), dtype=np.uint8)
        trimap = _strip_trimap(200, 160)

        alpha, fg = matting_module.run_matting(img, trimap)

        y0, _, _, _ = matting_module.solve_window(trimap)
        assert y0 > 4  # the window really does leave part of the ROI out
        assert (alpha[:4] == 0.0).all()
        # known pixels outside the window keep the source colour so compositing is lossless
        assert np.abs(fg[:4] * 255 - img[:4]).max() < 1.0

    def test_known_foreground_stays_opaque(self):
        rng = np.random.default_rng(9)
        img = rng.integers(0, 256, size=(80, 80, 3), dtype=np.uint8)
        trimap = _strip_trimap(80, 80)

        alpha, _ = matting_module.run_matting(img, trimap)

        assert (alpha[trimap == 255] > 0.99).all()

    def test_window_solve_matches_a_full_frame_solve(self):
        rng = np.random.default_rng(7)
        img = rng.integers(90, 160, size=(160, 240, 3), dtype=np.uint8)
        trimap = _strip_trimap(160, 240)

        alpha, _ = matting_module.run_matting(img, trimap)
        rgb = matting_module.cv2.cvtColor(img, matting_module.cv2.COLOR_BGR2RGB).astype(np.float64) / 255
        expected = matting_module.estimate_alpha_cf(rgb, trimap.astype(np.float64) / 255.0)

        unknown = trimap == 128
        assert np.abs(alpha[unknown] - expected[unknown]).max() < 0.05

    def test_no_solve_when_there_is_nothing_unknown(self, monkeypatch):
        def explode(*_args, **_kwargs):
            raise AssertionError("the solver must not run without unknown pixels")

        monkeypatch.setattr(matting_module, "estimate_alpha_cf", explode)
        img = np.full((16, 16, 3), 120, np.uint8)
        trimap = np.zeros((16, 16), np.uint8)
        trimap[2:6, 2:6] = 255

        alpha, fg = matting_module.run_matting(img, trimap)

        assert (alpha[2:6, 2:6] == 1.0).all()
        assert alpha.sum() == 16.0
        assert np.abs(fg * 255 - img).max() < 1.0

    def test_a_window_over_the_budget_is_solved_in_tiles(self, monkeypatch):
        seen = _spy_on_solver(monkeypatch)
        rng = np.random.default_rng(8)
        img = rng.integers(0, 256, size=(200, 400, 3), dtype=np.uint8)
        trimap = np.zeros((200, 400), np.uint8)
        trimap[10:190, 10:390] = 128
        trimap[60:140, 60:340] = 255
        budget = 20_000

        alpha, fg = matting_module.run_matting(img, trimap, max_solve_pixels=budget)

        assert len(seen) > 1
        assert all(h * w <= budget for h, w in seen)
        assert alpha.shape == (200, 400)
        assert fg.shape == (200, 400, 3)
        # the full resolution trimap still decides the known pixels
        assert (alpha[trimap == 255] > 0.99).all()
        assert (alpha[trimap == 0] < 0.01).all()

    def test_a_tiled_solve_stays_close_to_a_single_solve(self):
        rng = np.random.default_rng(11)
        img = rng.integers(90, 160, size=(160, 240, 3), dtype=np.uint8)
        trimap = _strip_trimap(160, 240)

        tiled, _ = matting_module.run_matting(img, trimap, max_solve_pixels=4_000)
        single, _ = matting_module.run_matting(img, trimap, max_solve_pixels=0)

        unknown = trimap == 128
        assert np.abs(tiled[unknown] - single[unknown]).mean() < 0.05

    def test_tiles_without_unknown_pixels_are_not_solved(self, monkeypatch):
        seen = _spy_on_solver(monkeypatch)
        rng = np.random.default_rng(12)
        img = rng.integers(0, 256, size=(300, 300, 3), dtype=np.uint8)
        trimap = np.zeros((300, 300), np.uint8)
        trimap[0:12, 0:12] = 128
        trimap[2:8, 2:8] = 255
        trimap[297:300, 297:300] = 255  # known product far away: the window covers everything

        matting_module.run_matting(img, trimap, max_solve_pixels=10_000)

        assert len(seen) == 1  # only the tile holding the unknown corner

    def test_a_tile_the_solver_cannot_take_falls_back_to_the_trimap(self, monkeypatch):
        seen = _spy_on_solver(monkeypatch)
        img = np.full((120, 120, 3), 100, np.uint8)
        trimap = np.full((120, 120), 255, np.uint8)
        trimap[50:70, 50:70] = 128  # unknown enclosed by foreground: no background context

        alpha, _ = matting_module.run_matting(img, trimap, max_solve_pixels=2_500)

        assert seen == []
        assert (alpha == 1.0).all()

    def test_pixel_budget_comes_from_the_environment(self, monkeypatch):
        monkeypatch.delenv("MATTE_MAX_SOLVE_PIXELS", raising=False)
        assert matting_module.max_solve_pixels() == matting_module.DEFAULT_MAX_SOLVE_PIXELS
        monkeypatch.setenv("MATTE_MAX_SOLVE_PIXELS", "12345")
        assert matting_module.max_solve_pixels() == 12345
        monkeypatch.setenv("MATTE_MAX_SOLVE_PIXELS", "0")
        assert matting_module.max_solve_pixels() is None
