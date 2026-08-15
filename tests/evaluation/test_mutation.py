"""Interpolation experiments, including a bit-exact check against production code."""

from __future__ import annotations

import numpy as np
import pytest

from backend import lash_extraction as pipeline
from backend.lash_extraction import matting as matting_module
from evaluation import metrics, mutation


@pytest.fixture
def rgba() -> np.ndarray:
    """Opaque dark square on transparent pixels that carry a loud colour.

    `estimate_foreground_ml` leaves arbitrary colours where alpha is 0, so the loud
    colour is not a contrived case: it is what the pipeline actually hands to warpAffine.
    """
    rng = np.random.default_rng(11)
    out = np.zeros((40, 40, 4), np.uint8)
    out[:, :, :3] = (0, 0, 255)  # BGR red everywhere, including alpha = 0 pixels
    # a textured interior: a flat colour would hide interpolation entirely
    out[12:28, 12:28, :3] = rng.integers(20, 90, (16, 16, 3), dtype=np.uint8)
    out[12:28, 12:28, 3] = 255
    return out


class TestWarpProduct:
    def test_identity_keeps_every_pixel(self, rgba):
        identity = np.array([[1, 0, 0], [0, 1, 0]], np.float64)
        out = mutation.warp_product(rgba, identity, (40, 40), "linear")
        assert np.array_equal(out, rgba)

    def test_nearest_with_an_integer_shift_is_lossless(self, rgba):
        matrix = np.array([[1, 0, 3], [0, 1, -2]], np.float64)
        out = mutation.warp_product(rgba, matrix, (40, 40), "nearest")
        inner = out[10:20, 20:30, :3]
        assert np.array_equal(inner, rgba[12:22, 17:27, :3])

    def test_fractional_shift_mutates_colours(self, rgba):
        matrix = np.array([[1, 0, 0.5], [0, 1, 0.0]], np.float64)
        out = mutation.warp_product(rgba, matrix, (40, 40), "linear")
        opaque = out[:, :, 3] >= 230
        source = mutation.nearest_source_colors(rgba, matrix, (40, 40))
        result = metrics.pixel_mutation(source, out[:, :, :3], opaque)
        assert result["exact_color_preservation_rate"] < 1.0

    def test_premultiplying_stops_transparent_colour_bleeding(self, rgba):
        """The landmark recompose path warps RGBA without premultiplying (matting.py)."""
        matrix = np.array([[1, 0, 0.5], [0, 1, 0.5]], np.float64)
        raw = mutation.warp_product(rgba, matrix, (40, 40), "linear", premultiply=False)
        premultiplied = mutation.warp_product(rgba, matrix, (40, 40), "linear", premultiply=True)
        fringe = (raw[:, :, 3] > 20) & (raw[:, :, 3] < 235)
        assert fringe.any()
        assert raw[:, :, 2][fringe].mean() > premultiplied[:, :, 2][fringe].mean() + 20


class TestMatchesProduction:
    def test_reproduces_recompose_onto_bit_for_bit(self, synthetic_landmarks, monkeypatch, rgba):
        """If this ever fails, the benchmark has stopped measuring the shipped code."""
        monkeypatch.setattr(matting_module, "detect_landmarks", lambda _img: synthetic_landmarks)
        roi = pipeline.EyeRoi(7, 11, 47, 51, 1.0)
        edited = np.full((80, 90, 3), 40, np.uint8)
        expected = pipeline.recompose_onto(rgba, roi, synthetic_landmarks, edited)

        matrix = mutation.recompose_matrix(roi, synthetic_landmarks, synthetic_landmarks)
        warped = mutation.warp_product(rgba, matrix, (90, 80), "linear", premultiply=False)
        assert np.array_equal(mutation.composite_over(warped, edited), expected)


class TestTransformMatrix:
    @pytest.mark.parametrize("name", list(mutation.TRANSFORMS))
    def test_every_named_transform_builds_a_matrix(self, name):
        matrix = mutation.transform_matrix(name, (40, 40))
        assert matrix.shape == (2, 3)

    def test_identity_is_the_identity(self):
        assert np.array_equal(mutation.transform_matrix("identity", (10, 10)), np.eye(2, 3))

    def test_unknown_transform_is_rejected(self):
        with pytest.raises(ValueError):
            mutation.transform_matrix("teleport", (10, 10))


class TestRunExperiment:
    def test_reports_every_variant_for_every_transform(self):
        from evaluation import backgrounds
        from evaluation.products import ProceduralProduct

        background = backgrounds.procedural_backgrounds(1, width=160, height=120, seed=1)[0]
        product = ProceduralProduct(seed=3, n_strands=12)
        rows = mutation.run_experiment(background, product, transforms=("identity", "rotate_10"))
        assert {row["transform"] for row in rows} == {"identity", "rotate_10"}
        assert {row["variant"] for row in rows} == set(mutation.VARIANTS)
        for row in rows:
            assert 0.0 <= row["exact_color_preservation_rate"] <= 1.0
            assert row["rgb_mae"] >= 0.0
            assert row["alpha_mad"] >= 0.0

    def test_identity_preserves_pixels_exactly(self):
        from evaluation import backgrounds
        from evaluation.products import ProceduralProduct

        background = backgrounds.procedural_backgrounds(1, width=160, height=120, seed=1)[0]
        rows = mutation.run_experiment(background, ProceduralProduct(seed=3, n_strands=12), ("identity",))
        for row in rows:
            assert row["exact_color_preservation_rate"] == pytest.approx(1.0)

    def test_rotation_is_lossless_for_nearest_and_lossy_for_linear(self):
        from evaluation import backgrounds
        from evaluation.products import ProceduralProduct

        background = backgrounds.procedural_backgrounds(1, width=160, height=120, seed=1)[0]
        rows = mutation.run_experiment(background, ProceduralProduct(seed=3, n_strands=12), ("rotate_10",))
        by_variant = {row["variant"]: row for row in rows}
        assert by_variant["nearest"]["exact_color_preservation_rate"] == pytest.approx(1.0)
        assert by_variant["linear"]["exact_color_preservation_rate"] < 0.9
        # nearest keeps colours but damages the geometry of 1-2 px strands
        assert by_variant["nearest"]["alpha_grad"] > by_variant["premultiplied_linear"]["alpha_grad"]


def test_gaussian_pyramid_of_roi_downscale_is_measured():
    """crop_roi shrinks any ROI wider than MAX_ROI_WIDTH, before extraction even starts."""
    rng = np.random.default_rng(0)
    image = rng.integers(0, 255, (200, 400, 3), dtype=np.uint8)
    rows = mutation.roi_downscale_experiment(image, widths=(400, 200))
    assert [row["roi_width"] for row in rows] == [400, 200]
    assert rows[0]["exact_color_preservation_rate"] == pytest.approx(1.0)  # no resize at all
    assert rows[0]["interpolation"] == "none"
    assert rows[1]["exact_color_preservation_rate"] < 1.0
    assert rows[1]["rgb_mae"] > 0.0
