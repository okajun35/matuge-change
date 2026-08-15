"""Eye region of interest: the coordinate space every extraction step works in."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from backend.lash_extraction.landmarks import LEFT_EYE, RIGHT_EYE

MAX_ROI_WIDTH = 1100


@dataclass
class EyeRoi:
    x0: int
    y0: int
    x1: int
    y1: int
    scale: float  # roi image = crop resized by this factor


def compute_eye_roi(landmarks: np.ndarray, img_shape: tuple) -> EyeRoi:
    h, w = img_shape[:2]
    eye_pts = landmarks[LEFT_EYE + RIGHT_EYE]
    ex0, ey0 = eye_pts.min(axis=0)
    ex1, ey1 = eye_pts.max(axis=0)
    ew, eh = ex1 - ex0, ey1 - ey0
    # generous margins: lashes extend up/outward well beyond eye contour
    x0 = int(max(0, ex0 - 0.45 * ew))
    x1 = int(min(w, ex1 + 0.45 * ew))
    y0 = int(max(0, ey0 - 2.4 * eh))
    y1 = int(min(h, ey1 + 1.4 * eh))
    scale = min(1.0, MAX_ROI_WIDTH / max(1, x1 - x0))
    return EyeRoi(x0, y0, x1, y1, scale)


def crop_roi(img: np.ndarray, roi: EyeRoi) -> np.ndarray:
    crop = img[roi.y0 : roi.y1, roi.x0 : roi.x1]
    if roi.scale < 1.0:
        crop = cv2.resize(crop, None, fx=roi.scale, fy=roi.scale, interpolation=cv2.INTER_AREA)
    return crop
