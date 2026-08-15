"""Feeds every case through the production pipeline and scores the result.

Each case is run in up to four configurations, because they answer different questions:

| mode        | brush  | question                                                |
| ----------- | ------ | ------------------------------------------------------- |
| bare        | auto   | difference-based extraction, no user correction          |
| worn_only   | auto   | darkness fallback, no user correction                    |
| bare        | oracle | how much a perfect three-value brush could recover      |
| worn_only   | oracle | same, without a bare photo                              |

Everything is compared in image coordinates: an ROI that misses part of the product must
show up as lost recall, not be silently excluded.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np

from evaluation import metrics
from evaluation.dataset import Case
from evaluation.pipeline import (
    composite_on,
    oracle_constraints,
    run_pipeline,
    to_image_space,
    to_roi_space,
)

MODES = ("bare", "worn_only")
BRUSHES = ("auto", "oracle")
RECOMPOSITION_BAND_PX = 5


@dataclass(frozen=True)
class RunConfig:
    modes: tuple[str, ...] = MODES
    brushes: tuple[str, ...] = BRUSHES
    fg_thresh: float = 0.70
    bg_thresh: float = 0.18
    unknown_band_px: int = 6
    boundary_tolerance: int = 2
    save_images: bool = True
    save_comparison: bool = True
    metadata_columns: tuple[str, ...] = field(
        default_factory=lambda: (
            "condition",
            "condition_value",
            "background",
            "product",
            "scale",
            "rotation_deg",
            "offset_x",
            "offset_y",
            "flip",
            "brightness",
            "contrast",
            "blur_sigma",
            "noise_sigma",
            "jpeg_quality",
            "bare_misalign_px",
            "shadow",
            "has_landmarks",
        )
    )


def _checker(shape: tuple[int, int], size: int = 8) -> np.ndarray:
    height, width = shape
    ys, xs = np.mgrid[0:height, 0:width]
    tile = ((ys // size + xs // size) % 2).astype(np.uint8)
    return (tile * 40 + 190).astype(np.uint8)[..., None].repeat(3, axis=2)


def _gray_to_bgr(single: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(single, cv2.COLOR_GRAY2BGR)


def comparison_image(
    case: Case,
    alpha_full: np.ndarray,
    product_full: np.ndarray,
    recomposed_full: np.ndarray,
    title: str = "",
) -> np.ndarray:
    """Bare | worn | GT mask | predicted alpha | product | recomposed | difference."""
    shape = case.gt_alpha.shape[:2]
    bare = case.bare.copy() if case.bare is not None else np.zeros((*shape, 3), np.uint8)
    alpha_view = _gray_to_bgr((np.clip(alpha_full, 0, 1) * 255).astype(np.uint8))
    product_alpha = product_full[:, :, 3:4].astype(np.float32) / 255.0
    product_view = (product_full[:, :, :3] * product_alpha + _checker(shape) * (1 - product_alpha)).astype(
        np.uint8
    )
    difference = cv2.convertScaleAbs(cv2.absdiff(recomposed_full, case.worn), alpha=4.0)
    tiles = [
        bare,
        case.worn.copy(),
        _gray_to_bgr(case.gt_mask),
        alpha_view,
        product_view,
        recomposed_full.copy(),
        difference,
    ]
    labels = ["bare", "worn", "gt mask", "pred alpha", "product", "recomposed", "diff x4"]
    for tile, label in zip(tiles, labels, strict=True):
        text = f"{label}  {title}" if label == "bare" else label
        cv2.putText(tile, text, (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 2)
        cv2.putText(tile, text, (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    return np.hstack(tiles)


def evaluate_case(
    case: Case,
    config: RunConfig = RunConfig(),
    output_root: str | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    gt_alpha = case.gt_alpha
    gt_mask = case.gt_mask > 0
    ignore = case.gt_ignore
    gt_product = case.gt_product
    shape = gt_alpha.shape[:2]

    for mode in config.modes:
        bare = case.bare if mode == "bare" else None
        if mode == "bare" and bare is None:
            continue
        for brush in config.brushes:
            constraints = None
            if brush == "oracle":
                # strokes live in ROI space, which only exists once the session is created
                constraints = lambda roi: oracle_constraints(to_roi_space(gt_alpha, roi))  # noqa: E731
            try:
                prediction = run_pipeline(
                    case.worn,
                    bare,
                    case.roi_rect,
                    constraints,
                    config.fg_thresh,
                    config.bg_thresh,
                    config.unknown_band_px,
                )
            except Exception as error:  # a failure is a result, not a reason to stop
                rows.append(_failure_row(case, mode, brush, error, config))
                continue
            alpha_full = to_image_space(prediction.alpha, prediction.roi, shape)
            product_full = to_image_space(prediction.product_rgba, prediction.roi, shape)
            base = case.bare if case.bare is not None else case.worn
            recomposed_full = to_image_space(
                composite_on(prediction, to_roi_space(base, prediction.roi)), prediction.roi, shape
            )

            band = cv2.dilate(
                case.gt_mask, np.ones((RECOMPOSITION_BAND_PX, RECOMPOSITION_BAND_PX), np.uint8)
            ).astype(bool)
            inside_roi = np.zeros(shape, bool)
            inside_roi[prediction.roi.y0 : prediction.roi.y1, prediction.roi.x0 : prediction.roi.x1] = True

            row: dict[str, Any] = {
                "case_id": case.case_id,
                "mode": mode,
                "brush": brush,
                "failed": False,
                "error": "",
                "seconds": round(prediction.seconds, 3),
                "roi_scale": prediction.roi.scale,
                "roi_rect": [prediction.roi.x0, prediction.roi.y0, prediction.roi.x1, prediction.roi.y1],
                "reconstruction_error": prediction.reconstruction_error,
                **{key: case.metadata.get(key) for key in config.metadata_columns},
                **metrics.segmentation(alpha_full, gt_mask),
                **{
                    f"{key}_ex_own": value
                    for key, value in metrics.segmentation(alpha_full, gt_mask, ignore=ignore).items()
                },
                **metrics.matting_errors(alpha_full, gt_alpha),
                **metrics.boundary(alpha_full, gt_mask, tolerance=config.boundary_tolerance),
                "component_delta": metrics.component_delta(alpha_full, gt_mask),
            }
            if gt_product is not None:
                row.update(
                    metrics.product_fidelity(
                        product_full[:, :, :3],
                        gt_product[:, :, :3],
                        product_full[:, :, 3],
                        gt_product[:, :, 3],
                    )
                )
            row.update(
                {
                    f"recompose_{key}": value
                    for key, value in metrics.region_fidelity(
                        recomposed_full, case.worn, band & inside_roi
                    ).items()
                }
            )
            rows.append(row)

            if output_root is not None and config.save_images:
                _save_case_outputs(
                    output_root, case, f"{mode}_{brush}", alpha_full, product_full, recomposed_full, config
                )
    return rows


def _failure_row(case: Case, mode: str, brush: str, error: Exception, config: RunConfig) -> dict[str, Any]:
    """A run that raised. Recorded with NaN metrics so it cannot flatter an average.

    The pipeline raises for instance when the trimap ends up without any foreground
    (`pymatting.trimap_split`), which is what happens when the probability map never
    reaches `fg_thresh`.
    """
    return {
        "case_id": case.case_id,
        "mode": mode,
        "brush": brush,
        "failed": True,
        "error": f"{type(error).__name__}: {error}",
        **{key: case.metadata.get(key) for key in config.metadata_columns},
        **dict.fromkeys(
            (
                "dice",
                "iou",
                "precision",
                "recall",
                "dice_ex_own",
                "iou_ex_own",
                "precision_ex_own",
                "recall_ex_own",
                "mad",
                "sad",
                "mse",
                "grad",
                "boundary_f1",
                "boundary_precision",
                "boundary_recall",
                "rgb_mae",
                "rgb_rmse",
                "recompose_mae",
                "recompose_rmse",
                "reconstruction_error",
            ),
            float("nan"),
        ),
    }


def _save_case_outputs(
    output_root: str,
    case: Case,
    tag: str,
    alpha_full: np.ndarray,
    product_full: np.ndarray,
    recomposed_full: np.ndarray,
    config: RunConfig,
) -> None:
    directory = os.path.join(output_root, "cases", case.case_id)
    os.makedirs(directory, exist_ok=True)
    cv2.imwrite(
        os.path.join(directory, f"predicted_alpha_{tag}.png"),
        (np.clip(alpha_full, 0, 1) * 255).astype(np.uint8),
    )
    cv2.imwrite(os.path.join(directory, f"predicted_product_{tag}.png"), product_full)
    cv2.imwrite(os.path.join(directory, "gt_mask.png"), case.gt_mask)
    if config.save_comparison:
        cv2.imwrite(
            os.path.join(directory, f"comparison_{tag}.png"),
            comparison_image(case, alpha_full, product_full, recomposed_full, title=tag),
        )


def run_dataset(
    cases: Iterable[Case],
    config: RunConfig = RunConfig(),
    output_root: str | None = None,
    progress: Callable[[str], None] = lambda _message: None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, case in enumerate(cases, start=1):
        progress(f"[{index}] {case.case_id} ({case.condition})")
        rows.extend(evaluate_case(case, config, output_root))
    return rows
