"""Lash extraction pipeline: landmarks, alignment, difference, probability, trimap, matting."""

from __future__ import annotations

import os
from dataclasses import dataclass

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
from pymatting import estimate_alpha_cf, estimate_foreground_ml

MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "models",
    "face_landmarker.task",
)

LEFT_EYE = [263, 249, 390, 373, 374, 380, 381, 382, 362, 466, 388, 387, 386, 385, 384, 398]
RIGHT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 246, 161, 160, 159, 158, 157, 173]
ALIGN_POINTS = LEFT_EYE + RIGHT_EYE + [6, 168, 197, 195, 5, 4, 1, 2, 98, 327]

MAX_ROI_WIDTH = 1100

_landmarker = None


def get_landmarker() -> vision.FaceLandmarker:
    global _landmarker
    if _landmarker is None:
        base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
        options = vision.FaceLandmarkerOptions(base_options=base_options, num_faces=1)
        _landmarker = vision.FaceLandmarker.create_from_options(options)
    return _landmarker


def detect_landmarks(bgr: np.ndarray) -> np.ndarray | None:
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = get_landmarker().detect(mp_image)
    if not result.face_landmarks:
        return None
    h, w = bgr.shape[:2]
    pts = np.array([[lm.x * w, lm.y * h] for lm in result.face_landmarks[0]], dtype=np.float64)
    return pts


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


def align_b_to_a(img_a: np.ndarray, lms_a: np.ndarray, img_b: np.ndarray, lms_b: np.ndarray) -> np.ndarray:
    """Warp image B (bare) onto image A (with product) using landmark affine + ECC refine."""
    src = lms_b[ALIGN_POINTS].astype(np.float32)
    dst = lms_a[ALIGN_POINTS].astype(np.float32)
    matrix, _ = cv2.estimateAffinePartial2D(src, dst, method=cv2.LMEDS)
    if matrix is None:
        matrix = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float64)
    h, w = img_a.shape[:2]
    warped = cv2.warpAffine(img_b, matrix, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return warped


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


def _norm_percentile(
    x: np.ndarray, lo: float = 50.0, hi: float = 99.5, min_range: float = 1e-3
) -> np.ndarray:
    """Percentile stretch to [0,1]. Inputs are on a 0-255 scale, so a spread below
    ``min_range`` carries no signal and is treated as flat (avoids amplifying noise)."""
    plo, phi = np.percentile(x, [lo, hi])
    if phi - plo < min_range:
        return np.zeros_like(x)
    return np.clip((x - plo) / (phi - plo), 0.0, 1.0)


def difference_map(roi_a: np.ndarray, roi_b: np.ndarray) -> np.ndarray:
    """Evidence map in [0,1]: where the worn image differs from the aligned bare image."""
    lab_a = cv2.cvtColor(roi_a, cv2.COLOR_BGR2LAB).astype(np.float32)
    lab_b = cv2.cvtColor(roi_b, cv2.COLOR_BGR2LAB).astype(np.float32)
    # lashes are dark: darkening of A relative to B is the strongest cue
    darkening = np.clip(lab_b[..., 0] - lab_a[..., 0], 0, None)
    chroma = np.linalg.norm(lab_a[..., 1:] - lab_b[..., 1:], axis=-1)
    ga = cv2.cvtColor(roi_a, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gb = cv2.cvtColor(roi_b, cv2.COLOR_BGR2GRAY).astype(np.float32)
    grad_a = cv2.magnitude(cv2.Sobel(ga, cv2.CV_32F, 1, 0), cv2.Sobel(ga, cv2.CV_32F, 0, 1))
    grad_b = cv2.magnitude(cv2.Sobel(gb, cv2.CV_32F, 1, 0), cv2.Sobel(gb, cv2.CV_32F, 0, 1))
    grad_diff = np.clip(grad_a - grad_b, 0, None)
    d = (
        0.6 * _norm_percentile(darkening)
        + 0.15 * _norm_percentile(chroma)
        + 0.25 * _norm_percentile(grad_diff)
    )
    return np.clip(d, 0.0, 1.0)


def darkness_map(roi_a: np.ndarray) -> np.ndarray:
    """Fallback evidence when no bare image exists: dark thin structures near the eye."""
    gray = cv2.cvtColor(roi_a, cv2.COLOR_BGR2GRAY).astype(np.float32)
    blur = cv2.GaussianBlur(gray, (0, 0), 8)
    darkness = np.clip(blur - gray, 0, None)  # locally dark structures
    return _norm_percentile(darkness)


def eye_prior(roi_shape: tuple, landmarks: np.ndarray, roi: EyeRoi) -> np.ndarray:
    """Soft prior around the eyes where lashes can plausibly exist."""
    h, w = roi_shape[:2]
    mask = np.zeros((h, w), np.uint8)
    for idx_set in (LEFT_EYE, RIGHT_EYE):
        pts = (landmarks[idx_set] - [roi.x0, roi.y0]) * roi.scale
        cv2.fillPoly(mask, [pts.astype(np.int32)], 255)
    eye_h = max(4, int(0.045 * w))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * eye_h + 1, 2 * eye_h + 1))
    dilated = cv2.dilate(mask, kernel, iterations=2)
    prior = cv2.GaussianBlur(dilated.astype(np.float32) / 255.0, (0, 0), eye_h * 0.8)
    return np.clip(prior * 1.4, 0.0, 1.0)


def initial_probability(evidence: np.ndarray, prior: np.ndarray) -> np.ndarray:
    return np.clip(evidence * prior, 0.0, 1.0)


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
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * unknown_band_px + 1, 2 * unknown_band_px + 1))
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
        rgba_roi,
        m_total,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0, 0),
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
