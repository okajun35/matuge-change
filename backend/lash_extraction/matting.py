"""Trimap building, closed-form matting and compositing."""

from __future__ import annotations

import cv2
import numpy as np
from pymatting import estimate_alpha_cf, estimate_foreground_ml

from backend.lash_extraction.landmarks import ALIGN_POINTS, detect_landmarks
from backend.lash_extraction.roi import EyeRoi


def build_trimap(
    prob: np.ndarray,
    constraints: np.ndarray | None,
    fg_thresh: float = 0.70,
    bg_thresh: float = 0.18,
    unknown_band_px: int = 6,
) -> np.ndarray:
    """Trimap uint8: 255 FG / 128 unknown / 0 BG.

    constraints: int8 map, +1 user product, 2 user unknown, -1 user background, 0 none.
    """
    fg = prob >= fg_thresh
    maybe = prob >= bg_thresh
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * unknown_band_px + 1, 2 * unknown_band_px + 1)
    )
    unknown = cv2.dilate(maybe.astype(np.uint8), kernel).astype(bool)
    trimap = np.zeros(prob.shape, np.uint8)
    trimap[unknown] = 128
    trimap[fg] = 255
    if constraints is not None:
        trimap[constraints == 2] = 128
        trimap[constraints == 1] = 255
        trimap[constraints == -1] = 0
    return trimap


def run_matting(roi_a: np.ndarray, trimap: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Closed-form matting + ML foreground estimation. Returns (alpha float, fg float BGR)."""
    img = cv2.cvtColor(roi_a, cv2.COLOR_BGR2RGB).astype(np.float64) / 255.0
    tri = trimap.astype(np.float64) / 255.0
    alpha = estimate_alpha_cf(img, tri)
    fg = estimate_foreground_ml(img, alpha)
    fg_bgr = cv2.cvtColor((fg * 255).astype(np.uint8), cv2.COLOR_RGB2BGR).astype(np.float64) / 255.0
    return alpha, fg_bgr


def composite(alpha: np.ndarray, fg_bgr: np.ndarray, base_bgr: np.ndarray) -> np.ndarray:
    a = alpha[..., None]
    out = a * fg_bgr + (1.0 - a) * (base_bgr.astype(np.float64) / 255.0)
    return (np.clip(out, 0, 1) * 255).astype(np.uint8)


def recompose_onto(
    rgba_roi: np.ndarray,
    roi: EyeRoi,
    lms_worn: np.ndarray,
    edited_bgr: np.ndarray,
) -> np.ndarray | None:
    """Composite an extracted ROI-space product RGBA onto an edited model image.

    Maps ROI coords -> worn-image coords -> edited-image coords (landmark affine),
    then alpha-blends. Returns None if no face is detected in the edited image.
    """
    lms_edit = detect_landmarks(edited_bgr)
    if lms_edit is None:
        return None
    src = lms_worn[ALIGN_POINTS].astype(np.float32)
    dst = lms_edit[ALIGN_POINTS].astype(np.float32)
    m_ae, _ = cv2.estimateAffinePartial2D(src, dst, method=cv2.LMEDS)
    if m_ae is None:
        return None
    inv_s = 1.0 / roi.scale
    m_roi_to_a = np.array([[inv_s, 0, roi.x0], [0, inv_s, roi.y0], [0, 0, 1]], dtype=np.float64)
    m_total = (np.vstack([m_ae, [0, 0, 1]]) @ m_roi_to_a)[:2]
    h, w = edited_bgr.shape[:2]
    warped = cv2.warpAffine(
        rgba_roi, m_total, (w, h),
        flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0),
    )
    alpha = warped[..., 3:4].astype(np.float64) / 255.0
    fg = warped[..., :3].astype(np.float64) / 255.0
    base = edited_bgr.astype(np.float64) / 255.0
    return (np.clip(alpha * fg + (1.0 - alpha) * base, 0, 1) * 255).astype(np.uint8)


def reconstruction_error(alpha: np.ndarray, fg_bgr: np.ndarray, roi_a: np.ndarray) -> float:
    """Mean abs error (0-255) inside alpha>0.05 when compositing back onto A itself."""
    recon = composite(alpha, fg_bgr, roi_a).astype(np.float32)
    mask = alpha > 0.05
    if not mask.any():
        return 0.0
    diff = np.abs(recon - roi_a.astype(np.float32))[mask]
    return float(diff.mean())
