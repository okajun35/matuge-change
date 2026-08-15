"""Runs the production extraction pipeline head-lessly, without changing it.

The benchmark drives `SessionService` on a throw-away store instead of going through
HTTP, because the API routes are thin wrappers around it: same code path, no server,
and nothing written into the repository's `data/`.

Two things this module deliberately provides:

* `oracle_constraints` — the brush strokes an ideal user would paint, derived from ground
  truth. The shipped product is human-in-the-loop (three-value brush), so scoring only
  the untouched automatic estimate would under-report it. Running both gives the floor
  (automatic) and the ceiling (perfect correction) of the same matting step.
* `to_image_space` — moves ROI-space results back into image coordinates so they can be
  compared with ground truth. When `roi.scale == 1.0` this is an exact paste, which is
  why the generator keeps images small enough to avoid `MAX_ROI_WIDTH` shrinking.
"""

from __future__ import annotations

import tempfile
import time
from collections.abc import Callable
from dataclasses import dataclass

import cv2
import numpy as np

from backend.lash_extraction import EyeRoi, composite, crop_roi
from backend.sessions.service import SessionService
from backend.sessions.store import SessionStore

CONSTRAINT_PRODUCT = 1
CONSTRAINT_UNKNOWN = 2
CONSTRAINT_BACKGROUND = -1


@dataclass(frozen=True)
class Prediction:
    """Everything the pipeline produced for one run, in ROI coordinates."""

    roi: EyeRoi
    roi_a: np.ndarray
    roi_b: np.ndarray | None
    alpha: np.ndarray  # float32 0..1
    product_rgba: np.ndarray  # BGRA uint8
    trimap: np.ndarray
    reconstruction_error: float
    mode: str
    seconds: float
    has_landmarks: bool


def run_pipeline(
    worn: np.ndarray,
    bare: np.ndarray | None,
    roi_rect: tuple[float, float, float, float] | None = None,
    constraints: np.ndarray | Callable[[EyeRoi], np.ndarray | None] | None = None,
    fg_thresh: float = 0.70,
    bg_thresh: float = 0.18,
    unknown_band_px: int = 6,
) -> Prediction:
    """Create a session and run matting exactly like `/api/session` + `/api/matte`.

    `constraints` may be a callable, because brush strokes live in ROI coordinates and
    the ROI is only known once the session exists.
    """
    started = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="lash-benchmark-") as root:
        service = SessionService(SessionStore(root))
        created = service.create(worn, bare, roi_rect)
        session_id = created["session_id"]
        roi = service.roi_of(session_id)
        strokes = constraints(roi) if callable(constraints) else constraints
        result = service.run_matte(session_id, strokes, fg_thresh, bg_thresh, unknown_band_px)
        store = service.store
        alpha = store.load_image(session_id, "alpha", flags=cv2.IMREAD_GRAYSCALE)
        return Prediction(
            roi=service.roi_of(session_id),
            roi_a=store.load_image(session_id, "roi_a"),
            roi_b=store.load_image(session_id, "roi_b") if store.has_layer(session_id, "roi_b") else None,
            alpha=alpha.astype(np.float32) / 255.0,
            product_rgba=store.load_image(session_id, "product_rgba", flags=cv2.IMREAD_UNCHANGED),
            trimap=store.load_image(session_id, "trimap", flags=cv2.IMREAD_GRAYSCALE),
            reconstruction_error=float(result["reconstruction_error"]),
            mode="bare" if bare is not None else "worn_only",
            seconds=time.perf_counter() - started,
            has_landmarks=store.has_array(session_id, "landmarks"),
        )


def to_roi_space(array: np.ndarray, roi: EyeRoi) -> np.ndarray:
    """Crop ground truth into ROI coordinates using the production crop."""
    return crop_roi(array, roi)


def to_image_space(array: np.ndarray, roi: EyeRoi, shape: tuple[int, int]) -> np.ndarray:
    """Paste an ROI-space result back into a full-size canvas (exact when scale == 1)."""
    height, width = shape[:2]
    canvas = np.zeros((height, width, *array.shape[2:]), array.dtype)
    target_w, target_h = roi.x1 - roi.x0, roi.y1 - roi.y0
    patch = array
    if (patch.shape[1], patch.shape[0]) != (target_w, target_h):
        patch = cv2.resize(patch, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
    canvas[roi.y0 : roi.y1, roi.x0 : roi.x1] = patch[:target_h, :target_w]
    return canvas


def oracle_constraints(
    gt_alpha_roi: np.ndarray,
    background_margin: int = 9,
    solid_threshold: int = 250,
    threshold: int = 128,
) -> np.ndarray | None:
    """Brush strokes a perfect user would paint, derived from ground truth.

    `+1` where the product is opaque, `-1` well outside it, `2` (force unknown) in the
    band between — the band users are told to leave to the matting step.

    The product region is *not* eroded: a lash is one or two pixels wide, so eroding it
    would delete the very thing the stroke is meant to assert, and a real brush cannot
    paint sub-pixel anyway. Returns None when the ROI holds no product at all, which
    leaves the automatic estimate untouched instead of handing matting an empty trimap.
    """
    solid = gt_alpha_roi >= solid_threshold
    if not solid.any():
        solid = gt_alpha_roi >= threshold
    if not solid.any():
        return None
    outer = cv2.dilate((gt_alpha_roi >= threshold).astype(np.uint8), _disc(background_margin))
    constraints = np.zeros(gt_alpha_roi.shape, np.int8)
    constraints[outer > 0] = CONSTRAINT_UNKNOWN
    constraints[outer == 0] = CONSTRAINT_BACKGROUND
    constraints[solid] = CONSTRAINT_PRODUCT
    return constraints


def _disc(radius: int) -> np.ndarray:
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))


def composite_on(prediction: Prediction, base_bgr: np.ndarray) -> np.ndarray:
    """Recompose the extracted product onto a base image with the production blend."""
    foreground = prediction.product_rgba[:, :, :3].astype(np.float64) / 255.0
    return composite(prediction.alpha.astype(np.float64), foreground, base_bgr)
