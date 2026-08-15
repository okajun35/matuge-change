import cv2
import numpy as np
import pytest

from backend.catalog.descriptor import DESCRIPTOR_DIM, LashDescriptor


def lash_alpha(width: int, height: int, thickness: int = 3, curl: float = 0.0) -> np.ndarray:
    """A synthetic lash-like alpha: a curved band across the image."""
    alpha = np.zeros((height, width), np.float32)
    xs = np.arange(width)
    ys = (height * 0.6 - curl * height * 0.3 * np.sin(np.pi * xs / max(1, width - 1))).astype(int)
    for x, y in zip(xs, ys, strict=False):
        cv2.circle(alpha, (int(x), int(y)), thickness, 1.0, -1)
    return cv2.GaussianBlur(alpha, (0, 0), 1.2)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


class TestLashDescriptor:
    def test_from_alpha_has_fixed_dimension(self):
        d = LashDescriptor.from_alpha(lash_alpha(80, 40))
        assert d.values.shape == (DESCRIPTOR_DIM,)
        assert d.values.dtype == np.float32

    def test_empty_alpha_is_zero_vector(self):
        d = LashDescriptor.from_alpha(np.zeros((32, 32), np.float32))
        assert d.is_empty
        assert not d.values.any()

    def test_scale_invariance(self):
        small = LashDescriptor.from_alpha(lash_alpha(80, 40, thickness=2))
        large = LashDescriptor.from_alpha(lash_alpha(160, 80, thickness=4))
        assert cosine(small.values, large.values) > 0.9

    def test_translation_invariance(self):
        base = lash_alpha(80, 40)
        shifted = np.zeros((60, 120), np.float32)
        shifted[10:50, 20:100] = base
        assert (
            cosine(
                LashDescriptor.from_alpha(base).values,
                LashDescriptor.from_alpha(shifted).values,
            )
            > 0.95
        )

    def test_different_shapes_are_less_similar_than_same_shape(self):
        straight = LashDescriptor.from_alpha(lash_alpha(80, 40, curl=0.0))
        curled = LashDescriptor.from_alpha(lash_alpha(80, 40, curl=1.0))
        same = LashDescriptor.from_alpha(lash_alpha(80, 40, curl=0.0))
        assert cosine(straight.values, curled.values) < cosine(straight.values, same.values)

    def test_to_list_is_json_serialisable_floats(self):
        values = LashDescriptor.from_alpha(lash_alpha(40, 20)).to_list()
        assert len(values) == DESCRIPTOR_DIM
        assert all(isinstance(v, float) for v in values)

    def test_rejects_non_2d_input(self):
        with pytest.raises(ValueError):
            LashDescriptor.from_alpha(np.zeros((8, 8, 3), np.float32))


class TestAlphaCoverage:
    def test_coverage_of_empty_alpha_is_zero(self):
        from backend.catalog.descriptor import alpha_coverage

        assert alpha_coverage(np.zeros((10, 10), np.float32)) == 0.0

    def test_coverage_counts_visible_pixels(self):
        from backend.catalog.descriptor import alpha_coverage

        alpha = np.zeros((10, 10), np.float32)
        alpha[:5] = 1.0
        assert alpha_coverage(alpha) == pytest.approx(0.5)
