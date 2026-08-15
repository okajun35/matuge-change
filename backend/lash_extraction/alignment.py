"""Aligning the bare image onto the worn image (landmark affine + ECC refine)."""

from __future__ import annotations

import cv2
import numpy as np

from backend.lash_extraction.landmarks import ALIGN_POINTS


def align_b_to_a(img_a: np.ndarray, lms_a: np.ndarray, img_b: np.ndarray, lms_b: np.ndarray) -> np.ndarray:
    """Warp image B (bare) onto image A (with product) using a landmark affine."""
    src = lms_b[ALIGN_POINTS].astype(np.float32)
    dst = lms_a[ALIGN_POINTS].astype(np.float32)
    matrix, _ = cv2.estimateAffinePartial2D(src, dst, method=cv2.LMEDS)
    if matrix is None:
        matrix = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float64)
    h, w = img_a.shape[:2]
    return cv2.warpAffine(img_b, matrix, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def ecc_refine(roi_a: np.ndarray, roi_b: np.ndarray) -> np.ndarray:
    """Refine alignment of roi_b onto roi_a within the eye ROI (euclidean ECC)."""
    ga = cv2.cvtColor(roi_a, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    gb = cv2.cvtColor(roi_b, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    warp = np.eye(2, 3, dtype=np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 100, 1e-5)
    try:
        _, warp = cv2.findTransformECC(ga, gb, warp, cv2.MOTION_EUCLIDEAN, criteria, None, 5)
        h, w = roi_a.shape[:2]
        return cv2.warpAffine(
            roi_b,
            warp,
            (w, h),
            flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
            borderMode=cv2.BORDER_REPLICATE,
        )
    except cv2.error:
        return roi_b
