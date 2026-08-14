"""Where are the lashes? Evidence maps and the resulting probability."""

from __future__ import annotations

import cv2
import numpy as np

from backend.lash_extraction.landmarks import LEFT_EYE, RIGHT_EYE
from backend.lash_extraction.roi import EyeRoi


def _norm_percentile(x: np.ndarray, lo: float = 50.0, hi: float = 99.5) -> np.ndarray:
    plo, phi = np.percentile(x, [lo, hi])
    # a spread this small is float noise, not signal: normalising it amplifies nothing
    if phi - plo < 1e-3:
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
