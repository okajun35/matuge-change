"""Extraction use cases: create a session, run matting, recompose onto an edited face."""

from __future__ import annotations

import logging
from collections.abc import Callable
from datetime import UTC, datetime
from math import isfinite
from typing import Any

import cv2
import numpy as np

from backend.jobs.memory import release_memory
from backend.lash_extraction import (
    EyeRoi,
    align_b_into_roi,
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
    product_bbox,
    recompose_onto,
    reconstruction_error,
    run_matting,
    solve_settings,
)
from backend.sessions.errors import FaceNotDetected, MatteNotReady
from backend.sessions.store import SessionStore

logger = logging.getLogger("backend.matte")

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
        """Start a session from images already held in memory."""
        return self.create_lazily(lambda: img_a, None if img_b is None else (lambda: img_b), roi_rect)

    def create_lazily(
        self,
        load_a: Callable[[], np.ndarray],
        load_b: Callable[[], np.ndarray] | None,
        roi_rect: tuple[float, float, float, float] | None = None,
    ) -> dict[str, Any]:
        """Start a session, decoding one source image at a time.

        `roi_rect` gives the eye region for images without a detectable face
        (profiles, eye close-ups); otherwise landmarks locate it. The worn image
        is persisted and released before the bare one is decoded: phone photos
        are ~36MB each as arrays, and holding both plus a full-size aligned copy
        is what pushes small hosts (512MB) over the limit.
        """
        session_id: str | None = None
        img_a = img_b = None
        try:
            img_a = load_a()
            # a crash during analysis is usually about the source size, so log it before detection
            logger.info("session create: worn image %dx%d", img_a.shape[1], img_a.shape[0])
            lms_a = detect_landmarks(img_a)
            if roi_rect is None and lms_a is None:
                raise FaceNotDetected("no face detected in the worn image")

            manual = roi_rect is not None
            roi = manual_eye_roi(roi_rect, img_a.shape) if manual else compute_eye_roi(lms_a, img_a.shape)
            roi_a = crop_roi(img_a, roi)

            session_id = self.store.create()
            # the uploads themselves are kept so a session can be re-derived later
            self.store.save_image(session_id, "source_with", img_a)
            img_a = None
            release_memory()

            roi_b = None
            if load_b is not None:
                img_b = load_b()
                lms_b = detect_landmarks(img_b)
                if lms_b is None and not manual:
                    raise FaceNotDetected("no face detected in the bare image")
                self.store.save_image(session_id, "source_without", img_b)
                aligned_roi = (
                    crop_roi(img_b, roi)
                    if lms_a is None or lms_b is None
                    else align_b_into_roi(img_b, lms_b, lms_a, roi)
                )
                img_b = None
                release_memory()
                roi_b = ecc_refine(roi_a, aligned_roi)
                evidence = difference_map(roi_a, roi_b)
            else:
                evidence = darkness_map(roi_a)

            # no landmarks -> no eye prior to restrict the evidence with
            prior = np.ones(roi_a.shape[:2]) if lms_a is None else eye_prior(roi_a.shape, lms_a, roi)
            prob = initial_probability(evidence, prior)
            logger.info(
                "session create: roi %dx%d scale=%.3f mode=%s",
                roi_a.shape[1],
                roi_a.shape[0],
                roi.scale,
                "manual" if manual else "auto",
            )
            return self._persist_created(
                session_id,
                roi,
                roi_a,
                roi_b,
                lms_a,
                evidence,
                prob,
                manual,
                load_b is not None,
            )
        except Exception:
            if session_id is not None:
                self.store.discard(session_id)
            raise
        finally:
            # a failed request must not leave a 12MP arena behind for the next one: the
            # raised exception keeps this frame alive, so the sources are dropped by hand
            img_a = img_b = None
            release_memory()

    def _persist_created(
        self,
        session_id: str,
        roi: EyeRoi,
        roi_a: np.ndarray,
        roi_b: np.ndarray | None,
        lms_a: np.ndarray | None,
        evidence: np.ndarray,
        prob: np.ndarray,
        manual: bool,
        has_bare_source: bool,
    ) -> dict[str, Any]:
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
            + (["source_without"] if has_bare_source else [])
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
        mode, budget = solve_settings()
        alpha, fg = run_matting(roi_a, trimap, mode=mode, max_solve_pixels=budget)
        report(85, "foreground")

        rgba = np.dstack([(fg * 255).astype(np.uint8), (alpha * 255).astype(np.uint8)])
        self.store.save_image(session_id, "trimap", trimap)
        self.store.save_gray(session_id, "alpha", alpha)
        self.store.save_image(session_id, "product_rgba", rgba)
        bbox = product_bbox(rgba)
        meta = self.store.load_meta(session_id)
        meta["product_bbox"] = [int(v) for v in bbox] if bbox is not None else None
        self.store.save_meta(session_id, meta)

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
                "solve_mode": mode,
                "max_solve_pixels": budget,
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
        angle: float = 0.0,
        flip: bool = False,
    ) -> dict[str, Any]:
        self.store.require(session_id)
        if not self.store.has_layer(session_id, "product_rgba"):
            raise MatteNotReady("run matting first")
        if not isfinite(angle) or not -180 <= angle <= 180:
            raise ValueError("angle must be finite and between -180 and 180 degrees")
        if dest_rect is None and (angle != 0 or flip):
            raise ValueError("angle and flip require dest_rect")
        rgba = self.store.load_image(session_id, "product_rgba", flags=-1)
        if dest_rect is not None:
            roi = manual_eye_roi(dest_rect, edited.shape)
            out = self._recompose_into_rect(rgba, edited, roi, angle, flip)
            self.store.save_image(session_id, "source_edited", edited)
            self.store.save_image(session_id, "composite_on_edited", out)
            meta = self.store.load_meta(session_id)
            normalized = [roi.x0, roi.y0, roi.x1, roi.y1]
            meta["dest_rect"] = normalized
            meta["dest_angle"] = float(angle)
            meta["dest_flip"] = bool(flip)
            bbox = product_bbox(rgba)
            meta["product_bbox"] = [int(v) for v in bbox] if bbox is not None else None
            self.store.save_meta(session_id, meta)
            return {
                "layers": ["composite_on_edited"],
                "dest_rect": normalized,
                "angle": float(angle),
                "flip": bool(flip),
                "product_bbox": meta["product_bbox"],
            }
        if not self.store.has_array(session_id, "landmarks"):
            raise FaceNotDetected(
                "this session has no face landmarks (manual ROI): recompose needs a detected face"
            )
        self.store.save_image(session_id, "source_edited", edited)
        lms_worn = self.store.load_array(session_id, "landmarks")
        out = recompose_onto(rgba, self.roi_of(session_id), lms_worn, edited)
        if out is None:
            raise FaceNotDetected("no face detected in the edited image")
        self.store.save_image(session_id, "composite_on_edited", out)
        return {"layers": ["composite_on_edited"]}

    @staticmethod
    def _recompose_into_rect(
        rgba: np.ndarray,
        edited: np.ndarray,
        roi: EyeRoi,
        angle: float = 0.0,
        flip: bool = False,
    ) -> np.ndarray:
        """Fit and similarity-transform RGBA product pixels into a destination rectangle."""
        bbox = product_bbox(rgba)
        if bbox is not None:
            x0, y0, x1, y1 = bbox
            rgba = rgba[y0:y1, x0:x1]
        if flip:
            rgba = cv2.flip(rgba, 1)
        target_w, target_h = roi.x1 - roi.x0, roi.y1 - roi.y0
        src_h, src_w = rgba.shape[:2]
        scale = min(target_w / src_w, target_h / src_h)
        fit_w = max(1, round(src_w * scale))
        fit_h = max(1, round(src_h * scale))

        color = rgba[:, :, :3].astype(np.float32)
        alpha = rgba[:, :, 3].astype(np.float32) / 255.0
        premultiplied = np.dstack((color * alpha[:, :, None], alpha))
        interpolation = cv2.INTER_AREA if scale < 1 else cv2.INTER_LANCZOS4
        transformed = cv2.resize(premultiplied, (fit_w, fit_h), interpolation=interpolation)
        radians = np.deg2rad(-angle)
        cos_a, sin_a = abs(np.cos(radians)), abs(np.sin(radians))
        rotated_w = max(1, int(np.ceil(fit_w * cos_a + fit_h * sin_a)))
        rotated_h = max(1, int(np.ceil(fit_w * sin_a + fit_h * cos_a)))
        if angle:
            center = (fit_w / 2, fit_h / 2)
            matrix = cv2.getRotationMatrix2D(center, -angle, 1.0)
            matrix[0, 2] += (rotated_w - fit_w) / 2
            matrix[1, 2] += (rotated_h - fit_h) / 2
            transformed = cv2.warpAffine(
                transformed,
                matrix,
                (rotated_w, rotated_h),
                flags=cv2.INTER_LANCZOS4,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0,
            )
        resized_alpha = np.clip(transformed[:, :, 3], 0, 1)
        resized_color = np.zeros_like(transformed[:, :, :3])
        np.divide(
            transformed[:, :, :3],
            resized_alpha[:, :, None],
            out=resized_color,
            where=resized_alpha[:, :, None] > 1e-6,
        )

        out = edited.astype(np.float32).copy()
        cx = (roi.x0 + roi.x1) / 2
        cy = (roi.y0 + roi.y1) / 2
        x = round(cx - transformed.shape[1] / 2)
        y = round(cy - transformed.shape[0] / 2)
        ox0, oy0 = max(0, x), max(0, y)
        ox1, oy1 = min(out.shape[1], x + transformed.shape[1]), min(out.shape[0], y + transformed.shape[0])
        if ox0 < ox1 and oy0 < oy1:
            sx0, sy0 = ox0 - x, oy0 - y
            sx1, sy1 = sx0 + ox1 - ox0, sy0 + oy1 - oy0
            background = out[oy0:oy1, ox0:ox1]
            a = resized_alpha[sy0:sy1, sx0:sx1, None]
            background[:] = resized_color[sy0:sy1, sx0:sx1] * a + background * (1 - a)
        return np.clip(out, 0, 255).astype(np.uint8)

    def roi_of(self, session_id: str) -> EyeRoi:
        meta = self.store.load_meta(session_id)
        x0, y0, x1, y1 = meta["roi"]
        return EyeRoi(x0, y0, x1, y1, meta["scale"])
