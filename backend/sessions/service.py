"""Extraction use cases: create a session, run matting, recompose onto an edited face."""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

import cv2
import numpy as np

from backend.lash_extraction import (
    EyeRoi,
    align_b_to_a,
    build_trimap,
    composite,
    compute_eye_roi,
    crop_roi,
    darkness_map,
    detect_landmarks,
    difference_map,
    ecc_refine,
    eye_prior,
    initial_probability,
    manual_eye_roi,
    recompose_onto,
    reconstruction_error,
    run_matting,
)
from backend.sessions.errors import FaceNotDetected, MatteNotReady
from backend.sessions.store import SessionStore

ProgressReporter = Callable[[int, str], None]


def _silent(progress: int, stage: str) -> None:
    """Default reporter for the synchronous endpoint."""


class SessionService:
    def __init__(self, store: SessionStore) -> None:
        self.store = store

    def create(
        self,
        img_a: np.ndarray,
        img_b: np.ndarray | None,
        roi_rect: tuple[float, float, float, float] | None = None,
    ) -> dict[str, Any]:
        """Start a session. `roi_rect` gives the eye region for images without a
        detectable face (profiles, eye close-ups); otherwise landmarks locate it."""
        lms_a = detect_landmarks(img_a)
        if roi_rect is None and lms_a is None:
            raise FaceNotDetected("no face detected in the worn image")

        manual = roi_rect is not None
        roi = manual_eye_roi(roi_rect, img_a.shape) if manual else compute_eye_roi(lms_a, img_a.shape)
        roi_a = crop_roi(img_a, roi)

        roi_b = None
        if img_b is not None:
            lms_b = detect_landmarks(img_b)
            if lms_b is None and not manual:
                raise FaceNotDetected("no face detected in the bare image")
            aligned = img_b if lms_a is None or lms_b is None else align_b_to_a(img_a, lms_a, img_b, lms_b)
            roi_b = ecc_refine(roi_a, crop_roi(aligned, roi))
            evidence = difference_map(roi_a, roi_b)
        else:
            evidence = darkness_map(roi_a)

        # no landmarks -> no eye prior to restrict the evidence with
        prior = np.ones(roi_a.shape[:2]) if lms_a is None else eye_prior(roi_a.shape, lms_a, roi)
        prob = initial_probability(evidence, prior)

        session_id = self.store.create()
        # the uploads themselves are kept so a session can be re-derived later
        self.store.save_image(session_id, "source_with", img_a)
        if img_b is not None:
            self.store.save_image(session_id, "source_without", img_b)
        self.store.save_image(session_id, "roi_a", roi_a)
        if roi_b is not None:
            self.store.save_image(session_id, "roi_b", roi_b)
        self.store.save_gray(session_id, "difference", evidence)
        self.store.save_gray(session_id, "probability", prob)
        self.store.save_array(session_id, "probability", prob)
        if lms_a is not None:
            self.store.save_array(session_id, "landmarks", lms_a)
        self.store.save_meta(
            session_id,
            {
                "roi": [roi.x0, roi.y0, roi.x1, roi.y1],
                "scale": roi.scale,
                "has_bare": roi_b is not None,
                "width": roi_a.shape[1],
                "height": roi_a.shape[0],
                "mode": "manual" if manual else "auto",
            },
        )

        layers = (
            ["roi_a"]
            + (["roi_b"] if roi_b is not None else [])
            + ["source_with"]
            + (["source_without"] if img_b is not None else [])
            + ["difference", "probability"]
        )
        return {
            "session_id": session_id,
            "width": roi_a.shape[1],
            "height": roi_a.shape[0],
            "has_bare": roi_b is not None,
            "mode": "manual" if manual else "auto",
            "layers": layers,
        }

    def probability_shape(self, session_id: str) -> tuple[int, int]:
        self.store.require(session_id)
        return tuple(self.store.load_array(session_id, "probability").shape)

    def run_matte(
        self,
        session_id: str,
        constraints: np.ndarray | None,
        fg_thresh: float = 0.70,
        bg_thresh: float = 0.18,
        unknown_band_px: int = 6,
        report: ProgressReporter = _silent,
    ) -> dict[str, Any]:
        self.store.require(session_id)
        roi_a = self.store.load_image(session_id, "roi_a")
        prob = self.store.load_array(session_id, "probability")

        report(20, "trimap")
        trimap = build_trimap(prob, constraints, fg_thresh, bg_thresh, unknown_band_px)
        report(45, "alpha")
        alpha, fg = run_matting(roi_a, trimap)
        report(85, "foreground")

        rgba = np.dstack([(fg * 255).astype(np.uint8), (alpha * 255).astype(np.uint8)])
        self.store.save_image(session_id, "trimap", trimap)
        self.store.save_gray(session_id, "alpha", alpha)
        self.store.save_image(session_id, "product_rgba", rgba)

        layers = ["trimap", "alpha", "product_rgba"]
        if self.store.has_layer(session_id, "roi_b"):
            roi_b = self.store.load_image(session_id, "roi_b")
            self.store.save_image(session_id, "composite_on_bare", composite(alpha, fg, roi_b))
            layers.append("composite_on_bare")

        error = reconstruction_error(alpha, fg, roi_a)
        self.store.append_run(
            session_id,
            {
                "created_at": datetime.now(UTC).isoformat(),
                "params": {
                    "fg_thresh": fg_thresh,
                    "bg_thresh": bg_thresh,
                    "unknown_band_px": unknown_band_px,
                },
                "reconstruction_error": error,
                "layers": layers,
            },
        )

        report(100, "done")
        return {"layers": layers, "reconstruction_error": error}

    def runs(self, session_id: str) -> list[dict[str, Any]]:
        self.store.require(session_id)
        return self.store.load_runs(session_id)

    def recompose(
        self,
        session_id: str,
        edited: np.ndarray,
        dest_rect: tuple[float, float, float, float] | None = None,
    ) -> dict[str, Any]:
        self.store.require(session_id)
        if not self.store.has_layer(session_id, "product_rgba"):
            raise MatteNotReady("run matting first")
        self.store.save_image(session_id, "source_edited", edited)
        rgba = self.store.load_image(session_id, "product_rgba", flags=-1)
        if dest_rect is not None:
            roi = manual_eye_roi(dest_rect, edited.shape)
            out = self._recompose_into_rect(rgba, edited, roi)
            self.store.save_image(session_id, "composite_on_edited", out)
            meta = self.store.load_meta(session_id)
            normalized = [roi.x0, roi.y0, roi.x1, roi.y1]
            meta["dest_rect"] = normalized
            self.store.save_meta(session_id, meta)
            return {"layers": ["composite_on_edited"], "dest_rect": normalized}
        if not self.store.has_array(session_id, "landmarks"):
            raise FaceNotDetected(
                "this session has no face landmarks (manual ROI): recompose needs a detected face"
            )
        lms_worn = self.store.load_array(session_id, "landmarks")
        out = recompose_onto(rgba, self.roi_of(session_id), lms_worn, edited)
        if out is None:
            raise FaceNotDetected("no face detected in the edited image")
        self.store.save_image(session_id, "composite_on_edited", out)
        return {"layers": ["composite_on_edited"]}

    @staticmethod
    def _recompose_into_rect(rgba: np.ndarray, edited: np.ndarray, roi: EyeRoi) -> np.ndarray:
        """Fit BGRA product pixels into a destination rectangle without halos."""
        target_w, target_h = roi.x1 - roi.x0, roi.y1 - roi.y0
        src_h, src_w = rgba.shape[:2]
        scale = min(target_w / src_w, target_h / src_h)
        fit_w = max(1, round(src_w * scale))
        fit_h = max(1, round(src_h * scale))

        color = rgba[:, :, :3].astype(np.float32)
        alpha = rgba[:, :, 3].astype(np.float32) / 255.0
        premultiplied = np.dstack((color * alpha[:, :, None], alpha))
        interpolation = cv2.INTER_AREA if scale < 1 else cv2.INTER_LANCZOS4
        resized = cv2.resize(premultiplied, (fit_w, fit_h), interpolation=interpolation)
        resized_alpha = np.clip(resized[:, :, 3], 0, 1)
        resized_color = np.zeros_like(resized[:, :, :3])
        np.divide(
            resized[:, :, :3],
            resized_alpha[:, :, None],
            out=resized_color,
            where=resized_alpha[:, :, None] > 1e-6,
        )

        out = edited.astype(np.float32).copy()
        x = roi.x0 + (target_w - fit_w) // 2
        y = roi.y0 + (target_h - fit_h) // 2
        background = out[y : y + fit_h, x : x + fit_w]
        a = resized_alpha[:, :, None]
        background[:] = resized_color * a + background * (1 - a)
        return np.clip(out, 0, 255).astype(np.uint8)

    def roi_of(self, session_id: str) -> EyeRoi:
        meta = self.store.load_meta(session_id)
        x0, y0, x1, y1 = meta["roi"]
        return EyeRoi(x0, y0, x1, y1, meta["scale"])
