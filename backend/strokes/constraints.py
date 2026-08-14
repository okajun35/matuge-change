"""Decoding the canvas overlay the browser paints into a trimap constraint map."""

from __future__ import annotations

import base64

import cv2
import numpy as np


def decode_constraints_png(data_url: str, shape: tuple[int, int]) -> np.ndarray:
    """Red = product (+1), green = unknown (2), blue = background (-1)."""
    raw = base64.b64decode(data_url.split(",")[-1])
    img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_UNCHANGED)
    if img is None or img.ndim != 3 or img.shape[:2] != tuple(shape):
        raise ValueError("invalid constraints image")

    painted = (img[..., 3] if img.shape[2] == 4 else np.full(shape, 255, np.uint8)) > 64
    b, g, r = (img[..., i].astype(int) for i in range(3))
    constraints = np.zeros(shape, np.int8)
    constraints[painted & (r >= g) & (r >= b)] = 1
    constraints[painted & (g > r) & (g >= b)] = 2
    constraints[painted & (b > r) & (b > g)] = -1
    return constraints
