"""Metric definitions.

Thresholds and conventions, stated once so the report can be read without guessing:

* a predicted alpha is binarised at **alpha >= 0.5**; ground truth masks are built the
  same way (`gt_mask = gt_alpha >= 128`).
* `ignore` is a boolean map of pixels excluded from *every* count. The runner uses it
  for the person's own lashes, which the pipeline is designed to pick up (its target is
  "the product plus the lashes it hides") but which are not part of the product.
* undefined ratios (no positives at all) are reported as NaN, never as 1.0 or 0.0, so
  they can be dropped from an average instead of quietly inflating it.
* IoU/Dice on a 1-2 px wide strand mostly measure the thick root band, so alpha-domain
  errors (`mad`, `sad`, `mse`, `grad`) are reported next to them. `grad` is the standard
  gradient error of the image matting literature and is the one that reacts to soft or
  smeared lash tips.
"""

from __future__ import annotations

import cv2
import numpy as np

BINARY_THRESHOLD = 0.5
GRADIENT_SIGMA = 1.4
FIDELITY_ALPHA_MIN = 0.9


def _as_alpha(alpha: np.ndarray) -> np.ndarray:
    out = alpha.astype(np.float32)
    if out.max() > 1.0:  # a uint8 alpha map
        out = out / 255.0
    return np.clip(out, 0.0, 1.0)


def _scored(shape: tuple[int, int], ignore: np.ndarray | None) -> np.ndarray:
    if ignore is None:
        return np.ones(shape, bool)
    return ~ignore.astype(bool)


def _ratio(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator > 0 else float("nan")


def segmentation(
    pred_alpha: np.ndarray,
    gt_mask: np.ndarray,
    ignore: np.ndarray | None = None,
    threshold: float = BINARY_THRESHOLD,
) -> dict[str, float]:
    """IoU / Dice / precision / recall of `pred_alpha >= threshold` against `gt_mask`."""
    scored = _scored(gt_mask.shape[:2], ignore)
    pred = (_as_alpha(pred_alpha) >= threshold) & scored
    truth = (np.asarray(gt_mask) > 0 if gt_mask.dtype != bool else gt_mask) & scored
    intersection = float(np.count_nonzero(pred & truth))
    union = float(np.count_nonzero(pred | truth))
    return {
        "iou": _ratio(intersection, union),
        "dice": _ratio(2 * intersection, float(pred.sum() + truth.sum())),
        "precision": _ratio(intersection, float(pred.sum())),
        "recall": _ratio(intersection, float(truth.sum())),
        "evaluated_px": float(np.count_nonzero(scored)),
        "pred_px": float(pred.sum()),
        "gt_px": float(truth.sum()),
    }


def _gradient_magnitude(alpha: np.ndarray) -> np.ndarray:
    dx = cv2.Sobel(cv2.GaussianBlur(alpha, (0, 0), GRADIENT_SIGMA), cv2.CV_32F, 1, 0)
    dy = cv2.Sobel(cv2.GaussianBlur(alpha, (0, 0), GRADIENT_SIGMA), cv2.CV_32F, 0, 1)
    return cv2.magnitude(dx, dy)


def matting_errors(
    pred_alpha: np.ndarray,
    gt_alpha: np.ndarray,
    ignore: np.ndarray | None = None,
) -> dict[str, float]:
    """Alpha-domain errors: MAD, SAD (in alpha-pixels), MSE and gradient error."""
    pred = _as_alpha(pred_alpha)
    truth = _as_alpha(gt_alpha)
    scored = _scored(truth.shape[:2], ignore)
    difference = np.abs(pred - truth)[scored]
    gradient = np.abs(_gradient_magnitude(pred) - _gradient_magnitude(truth))[scored]
    count = difference.size
    return {
        "mad": _ratio(float(difference.sum()), count),
        "sad": float(difference.sum()),
        "mse": _ratio(float((difference**2).sum()), count),
        "grad": float(gradient.sum() / 1000.0),
    }


def _boundary(mask: np.ndarray) -> np.ndarray:
    solid = mask.astype(np.uint8)
    eroded = cv2.erode(solid, np.ones((3, 3), np.uint8))
    return (solid - eroded).astype(bool)


def boundary(
    pred_alpha: np.ndarray,
    gt_mask: np.ndarray,
    tolerance: int = 2,
    threshold: float = BINARY_THRESHOLD,
    ignore: np.ndarray | None = None,
) -> dict[str, float]:
    """Boundary precision / recall / F1 with a pixel tolerance (Csurka et al.)."""
    scored = _scored(gt_mask.shape[:2], ignore)
    pred = (_as_alpha(pred_alpha) >= threshold) & scored
    truth = (np.asarray(gt_mask) > 0 if gt_mask.dtype != bool else gt_mask) & scored
    pred_edge, gt_edge = _boundary(pred), _boundary(truth)

    def within(edge: np.ndarray, other: np.ndarray) -> float:
        if not edge.any():
            return float("nan")
        if not other.any():
            return 0.0
        distance = cv2.distanceTransform((~other).astype(np.uint8), cv2.DIST_L2, 3)
        return float(np.count_nonzero(distance[edge] <= tolerance) / np.count_nonzero(edge))

    precision = within(pred_edge, gt_edge)
    recall = within(gt_edge, pred_edge)
    if np.isnan(precision) or np.isnan(recall) or precision + recall == 0:
        f1 = 0.0 if not np.isnan(recall) else float("nan")
    else:
        f1 = 2 * precision * recall / (precision + recall)
    return {"boundary_precision": precision, "boundary_recall": recall, "boundary_f1": f1}


def component_delta(
    pred_alpha: np.ndarray,
    gt_mask: np.ndarray,
    threshold: float = BINARY_THRESHOLD,
    min_area: int = 2,
) -> int:
    """Connected-component count difference: a cheap read on broken or merged strands."""

    def count(mask: np.ndarray) -> int:
        total, _, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
        return int(sum(1 for label in range(1, total) if stats[label, cv2.CC_STAT_AREA] >= min_area))

    truth = np.asarray(gt_mask) > 0 if gt_mask.dtype != bool else gt_mask
    return count(_as_alpha(pred_alpha) >= threshold) - count(truth)


def product_fidelity(
    pred_rgb: np.ndarray,
    gt_rgb: np.ndarray,
    pred_alpha: np.ndarray,
    gt_alpha: np.ndarray,
    alpha_min: float = FIDELITY_ALPHA_MIN,
) -> dict[str, float]:
    """RGB error of the extracted product where both alphas are confidently opaque.

    Foreground colour is genuinely undefined where alpha is partial (many colours
    composite to the same pixel), so only `alpha >= alpha_min` in *both* the prediction
    and the ground truth is compared.
    """
    mask = (_as_alpha(pred_alpha) >= alpha_min) & (_as_alpha(gt_alpha) >= alpha_min)
    if not mask.any():
        return {"rgb_mae": float("nan"), "rgb_rmse": float("nan"), "fidelity_px": 0.0}
    difference = pred_rgb[mask].astype(np.float64) - gt_rgb[mask].astype(np.float64)
    return {
        "rgb_mae": float(np.abs(difference).mean()),
        "rgb_rmse": float(np.sqrt((difference**2).mean())),
        "fidelity_px": float(np.count_nonzero(mask)),
    }


def region_fidelity(pred_bgr: np.ndarray, gt_bgr: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    """RGB error of a composited image inside `mask` (used around the product)."""
    if not mask.any():
        return {"mae": float("nan"), "rmse": float("nan"), "region_px": 0.0}
    difference = pred_bgr[mask].astype(np.float64) - gt_bgr[mask].astype(np.float64)
    return {
        "mae": float(np.abs(difference).mean()),
        "rmse": float(np.sqrt((difference**2).mean())),
        "region_px": float(np.count_nonzero(mask)),
    }


def pixel_mutation(before_rgb: np.ndarray, after_rgb: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    """How much a transform changed product pixels that were supposed to be preserved.

    `exact_color_preservation_rate` counts pixels whose three channels are bit-identical.
    Note that it can only reach 1.0 when no resampling happens at all; the magnitude
    metrics matter more for anything the production code actually does.
    """
    if not mask.any():
        return {
            "exact_color_preservation_rate": float("nan"),
            "rgb_mutation_rate": float("nan"),
            "rgb_mae": float("nan"),
            "rgb_max_error": float("nan"),
            "mutation_px": 0.0,
        }
    before = before_rgb[mask].astype(np.int32)
    after = after_rgb[mask].astype(np.int32)
    identical = np.all(before == after, axis=-1)
    difference = np.abs(after - before)
    return {
        "exact_color_preservation_rate": float(identical.mean()),
        "rgb_mutation_rate": float(1.0 - identical.mean()),
        "rgb_mae": float(difference.mean()),
        "rgb_max_error": float(difference.max()),
        "mutation_px": float(np.count_nonzero(mask)),
    }
