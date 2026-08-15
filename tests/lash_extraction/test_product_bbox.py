import numpy as np

from backend.lash_extraction import product_bbox


def test_product_bbox_returns_alpha_extent():
    rgba = np.zeros((100, 120, 4), np.uint8)
    rgba[30:70, 45:90, 3] = 255

    assert product_bbox(rgba) == (45, 30, 90, 70)


def test_product_bbox_drops_small_isolated_noise():
    rgba = np.zeros((100, 120, 4), np.uint8)
    rgba[30:70, 45:90, 3] = 255
    rgba[0, 0, 3] = 255

    assert product_bbox(rgba) == (45, 30, 90, 70)


def test_product_bbox_returns_none_for_empty_alpha():
    assert product_bbox(np.zeros((20, 30, 4), np.uint8)) is None


def test_product_bbox_applies_alpha_threshold():
    rgba = np.zeros((40, 50, 4), np.uint8)
    rgba[10:20, 15:30, 3] = 15

    assert product_bbox(rgba) is None
