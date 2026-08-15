"""Metrics are checked against analytically known answers, not against the pipeline."""

from __future__ import annotations

import numpy as np
import pytest

from evaluation import metrics


def _square(shape, y0, x0, size, value=1.0):
    out = np.zeros(shape, np.float32)
    out[y0 : y0 + size, x0 : x0 + size] = value
    return out


class TestSegmentation:
    def test_perfect_prediction(self):
        gt = _square((40, 40), 10, 10, 10)
        result = metrics.segmentation(gt, gt >= 0.5)
        assert result["iou"] == pytest.approx(1.0)
        assert result["dice"] == pytest.approx(1.0)
        assert result["precision"] == pytest.approx(1.0)
        assert result["recall"] == pytest.approx(1.0)

    def test_half_overlapping_squares(self):
        """10x10 squares offset by 5 px: intersection 50, union 150."""
        pred = _square((40, 40), 10, 10, 10)
        gt = _square((40, 40), 10, 15, 10) >= 0.5
        result = metrics.segmentation(pred, gt)
        assert result["iou"] == pytest.approx(50 / 150)
        assert result["dice"] == pytest.approx(2 * 50 / 200)
        assert result["precision"] == pytest.approx(0.5)
        assert result["recall"] == pytest.approx(0.5)

    def test_threshold_is_applied_to_the_alpha(self):
        pred = np.full((10, 10), 0.4, np.float32)
        gt = np.ones((10, 10), bool)
        assert metrics.segmentation(pred, gt, threshold=0.5)["recall"] == pytest.approx(0.0)
        assert metrics.segmentation(pred, gt, threshold=0.3)["recall"] == pytest.approx(1.0)

    def test_ignore_region_is_excluded_from_every_count(self):
        pred = _square((20, 20), 0, 0, 10)  # entirely a false positive
        gt = np.zeros((20, 20), bool)
        assert metrics.segmentation(pred, gt)["precision"] == pytest.approx(0.0)
        ignore = _square((20, 20), 0, 0, 10) > 0
        scored = metrics.segmentation(pred, gt, ignore=ignore)
        assert np.isnan(scored["precision"])  # nothing left to score
        assert scored["evaluated_px"] == 400 - 100

    def test_empty_prediction_and_empty_truth(self):
        empty = np.zeros((8, 8), np.float32)
        result = metrics.segmentation(empty, empty >= 0.5)
        assert np.isnan(result["iou"])  # undefined, must not be reported as a perfect 1.0


class TestMattingErrors:
    def test_identical_alpha_has_no_error(self):
        rng = np.random.default_rng(0)
        alpha = rng.random((32, 32)).astype(np.float32)
        result = metrics.matting_errors(alpha, alpha)
        assert result["mad"] == pytest.approx(0.0)
        assert result["mse"] == pytest.approx(0.0)
        assert result["grad"] == pytest.approx(0.0, abs=1e-6)

    def test_mad_and_sad_are_consistent(self):
        gt = np.zeros((10, 10), np.float32)
        pred = np.full((10, 10), 0.25, np.float32)
        result = metrics.matting_errors(pred, gt)
        assert result["mad"] == pytest.approx(0.25)
        assert result["sad"] == pytest.approx(25.0)
        assert result["mse"] == pytest.approx(0.0625)

    def test_gradient_error_catches_a_blurred_edge_that_mad_forgives(self):
        import cv2

        gt = _square((60, 60), 20, 20, 20)
        blurred = cv2.GaussianBlur(gt, (0, 0), 3.0)
        result = metrics.matting_errors(blurred, gt)
        assert result["mad"] < 0.05  # exactly why thin lashes need more than MAD/IoU
        assert result["grad"] > 5 * result["mad"]

    def test_respects_ignore(self):
        gt = np.zeros((10, 10), np.float32)
        pred = np.zeros((10, 10), np.float32)
        pred[0:5] = 1.0
        ignore = np.zeros((10, 10), bool)
        ignore[0:5] = True
        assert metrics.matting_errors(pred, gt, ignore=ignore)["mad"] == pytest.approx(0.0)


class TestBoundary:
    def test_identical_masks_score_one(self):
        gt = _square((40, 40), 10, 10, 12) >= 0.5
        result = metrics.boundary(gt.astype(np.float32), gt, tolerance=1)
        assert result["boundary_f1"] == pytest.approx(1.0)

    def test_tolerance_controls_how_strict_a_shift_is(self):
        gt = _square((40, 40), 10, 10, 12) >= 0.5
        shifted = _square((40, 40), 10, 13, 12)
        loose = metrics.boundary(shifted, gt, tolerance=3)["boundary_f1"]
        strict = metrics.boundary(shifted, gt, tolerance=1)["boundary_f1"]
        assert loose > strict
        assert 0.0 <= strict <= loose <= 1.0

    def test_missing_prediction_scores_zero_recall(self):
        gt = _square((30, 30), 5, 5, 10) >= 0.5
        result = metrics.boundary(np.zeros((30, 30), np.float32), gt, tolerance=2)
        assert result["boundary_recall"] == pytest.approx(0.0)


class TestComponentDelta:
    def test_counts_broken_strands(self):
        gt = np.zeros((20, 20), np.float32)
        gt[10, 2:18] = 1.0
        broken = gt.copy()
        broken[10, 9:11] = 0.0
        assert metrics.component_delta(gt, gt >= 0.5) == 0
        assert metrics.component_delta(broken, gt >= 0.5) == 1


class TestProductFidelity:
    def test_identical_product_has_no_error(self):
        rng = np.random.default_rng(1)
        rgb = rng.integers(0, 255, (16, 16, 3), dtype=np.uint8)
        alpha = np.ones((16, 16), np.float32)
        result = metrics.product_fidelity(rgb, rgb, alpha, alpha)
        assert result["rgb_mae"] == pytest.approx(0.0)
        assert result["rgb_rmse"] == pytest.approx(0.0)
        assert result["fidelity_px"] == 256

    def test_only_confidently_opaque_pixels_are_compared(self):
        gt_rgb = np.zeros((10, 10, 3), np.uint8)
        pred_rgb = np.zeros((10, 10, 3), np.uint8)
        pred_rgb[0:5] = 100  # wrong, but where alpha is low the estimate is undefined
        alpha = np.zeros((10, 10), np.float32)
        alpha[5:] = 1.0
        result = metrics.product_fidelity(pred_rgb, gt_rgb, alpha, alpha, alpha_min=0.9)
        assert result["rgb_mae"] == pytest.approx(0.0)
        assert result["fidelity_px"] == 50

    def test_reports_nan_when_nothing_is_comparable(self):
        rgb = np.zeros((8, 8, 3), np.uint8)
        alpha = np.zeros((8, 8), np.float32)
        assert np.isnan(metrics.product_fidelity(rgb, rgb, alpha, alpha)["rgb_mae"])


class TestRegionFidelity:
    def test_mae_and_rmse_on_a_known_difference(self):
        a = np.zeros((10, 10, 3), np.uint8)
        b = np.full((10, 10, 3), 4, np.uint8)
        mask = np.ones((10, 10), bool)
        result = metrics.region_fidelity(a, b, mask)
        assert result["mae"] == pytest.approx(4.0)
        assert result["rmse"] == pytest.approx(4.0)


class TestPixelMutation:
    def test_identical_pixels_are_fully_preserved(self):
        rng = np.random.default_rng(2)
        rgb = rng.integers(0, 255, (12, 12, 3), dtype=np.uint8)
        mask = np.ones((12, 12), bool)
        result = metrics.pixel_mutation(rgb, rgb, mask)
        assert result["exact_color_preservation_rate"] == pytest.approx(1.0)
        assert result["rgb_mutation_rate"] == pytest.approx(0.0)
        assert result["rgb_mae"] == pytest.approx(0.0)

    def test_a_single_changed_channel_counts_as_mutated(self):
        before = np.zeros((10, 10, 3), np.uint8)
        after = before.copy()
        after[0, 0, 2] = 1
        result = metrics.pixel_mutation(before, after, np.ones((10, 10), bool))
        assert result["rgb_mutation_rate"] == pytest.approx(1 / 100)
        assert result["rgb_max_error"] == 1

    def test_only_masked_pixels_are_judged(self):
        before = np.zeros((10, 10, 3), np.uint8)
        after = np.full((10, 10, 3), 50, np.uint8)
        mask = np.zeros((10, 10), bool)
        mask[0:2] = True
        after[0:2] = 0
        result = metrics.pixel_mutation(before, after, mask)
        assert result["exact_color_preservation_rate"] == pytest.approx(1.0)
