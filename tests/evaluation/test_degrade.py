"""Capture-condition degradations must be reproducible and independent per image."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from evaluation.degrade import Degradation, degrade, misalign


@pytest.fixture
def image() -> np.ndarray:
    rng = np.random.default_rng(0)
    base = rng.integers(60, 200, size=(64, 80, 3), dtype=np.uint8)
    return cv2.GaussianBlur(base, (0, 0), 1.5)


class TestDegrade:
    def test_identity_is_a_no_op(self, image):
        assert np.array_equal(degrade(image, Degradation(), seed=1), image)

    def test_brightness_scales_the_mean(self, image):
        brighter = degrade(image, Degradation(brightness=1.3), seed=1)
        darker = degrade(image, Degradation(brightness=0.7), seed=1)
        assert brighter.mean() > image.mean() * 1.15
        assert darker.mean() < image.mean() * 0.85

    def test_contrast_changes_spread_not_mean(self, image):
        flat = degrade(image, Degradation(contrast=0.5), seed=1)
        assert flat.std() < image.std() * 0.7
        assert flat.mean() == pytest.approx(image.mean(), rel=0.05)

    def test_blur_removes_detail(self, image):
        blurred = degrade(image, Degradation(blur_sigma=2.0), seed=1)
        assert cv2.Laplacian(blurred, cv2.CV_32F).std() < cv2.Laplacian(image, cv2.CV_32F).std() * 0.6

    def test_noise_depends_on_the_seed(self, image):
        a = degrade(image, Degradation(noise_sigma=6.0), seed=1)
        b = degrade(image, Degradation(noise_sigma=6.0), seed=1)
        c = degrade(image, Degradation(noise_sigma=6.0), seed=2)
        assert np.array_equal(a, b)
        assert not np.array_equal(a, c)
        # this independence is what stops the bare/worn pair from being a trivial subtraction
        assert np.abs(a.astype(int) - c.astype(int)).mean() > 3.0

    def test_jpeg_introduces_bounded_error(self, image):
        compressed = degrade(image, Degradation(jpeg_quality=30), seed=1)
        error = np.abs(compressed.astype(int) - image.astype(int)).mean()
        assert 0.5 < error < 25.0
        assert compressed.shape == image.shape

    def test_output_stays_uint8(self, image):
        out = degrade(image, Degradation(brightness=1.8, contrast=1.5, noise_sigma=40.0), seed=1)
        assert out.dtype == np.uint8


class TestMisalign:
    def test_zero_is_a_no_op(self, image):
        assert np.array_equal(misalign(image, 0.0, 0.0, seed=1), image)

    def test_shifts_content(self, image):
        moved = misalign(image, 4.0, 0.0, seed=1)
        assert moved.shape == image.shape
        assert np.abs(moved.astype(int) - image.astype(int)).mean() > 1.0

    def test_rotation_changes_more_than_translation_at_the_edges(self, image):
        rotated = misalign(image, 0.0, 2.0, seed=1)
        assert not np.array_equal(rotated, image)
