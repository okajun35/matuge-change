"""Helpers for locating the visible pixels of an extracted product."""

from __future__ import annotations

import cv2
import numpy as np


def product_bbox(
    rgba: np.ndarray,
    alpha_threshold: int = 16,
    min_area_ratio: float = 0.0005,
) -> tuple[int, int, int, int] | None:
    """Return the alpha foreground bbox, excluding tiny connected components."""
    if rgba.ndim != 3 or rgba.shape[2] < 4:
        raise ValueError("product_bbox expects an RGBA image")
    foreground = (rgba[:, :, 3] >= alpha_threshold).astype(np.uint8)
    count, _, stats, _ = cv2.connectedComponentsWithStats(foreground, connectivity=8)
    min_area = rgba.shape[0] * rgba.shape[1] * min_area_ratio
    boxes = []
    for label in range(1, count):
        x, y, width, height, area = stats[label]
        if area >= min_area:
            boxes.append((x, y, x + width, y + height))
    if not boxes:
        return None
    return (
        min(box[0] for box in boxes),
        min(box[1] for box in boxes),
        max(box[2] for box in boxes),
        max(box[3] for box in boxes),
    )
