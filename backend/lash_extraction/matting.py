"""Trimap building, closed-form matting and compositing."""

from __future__ import annotations

import logging
import os
import time

import cv2
import numpy as np
from pymatting import estimate_alpha_cf, estimate_foreground_ml

from backend.lash_extraction.landmarks import ALIGN_POINTS, detect_landmarks
from backend.lash_extraction.roi import EyeRoi

# Closed-form matting allocates a sparse Laplacian and several float64 buffers over the
# solved area, so its peak memory scales with the number of solved pixels (measured at
# roughly 3MB per 1000 pixels on top of a ~290MB baseline). 512MB hosts (Render's starter
# instance) cannot afford a whole-ROI solve, so they can opt into the tiled approximation.
# Quality comes first everywhere else: `full` is the default and is the solve this
# repository has always done.
FULL = "full"
TILED = "tiled"
SOLVE_MODES = (FULL, TILED)
SOLVE_MARGIN_PX = 32
TILE_OVERLAP_PX = 24
DEFAULT_MAX_SOLVE_PIXELS = 60_000

# Shared with `backend.observability` so the startup line and the per-run lines read as one
# stream: `MATTE_SOLVE_MODE` decides whether a host survives, so it has to be greppable.
logger = logging.getLogger("backend.matte")


def solve_mode() -> str:
    """`MATTE_SOLVE_MODE`, defaulting to the full-quality solve."""
    raw = os.environ.get("MATTE_SOLVE_MODE", "").strip().lower()
    if not raw:
        return FULL
    if raw not in SOLVE_MODES:
        raise ValueError(f"MATTE_SOLVE_MODE must be one of {SOLVE_MODES}, got {raw!r}")
    return raw


def max_solve_pixels() -> int | None:
    """Pixel budget for one tiled solve; `MATTE_MAX_SOLVE_PIXELS=0` means unbounded.

    Anything else that is not a pixel count is a configuration error rather than a silent
    fallback: reading a typo as "unbounded" would put a 512MB host back on the OOM path.
    """
    raw = os.environ.get("MATTE_MAX_SOLVE_PIXELS", "").strip()
    if not raw:
        return DEFAULT_MAX_SOLVE_PIXELS
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(
            f"MATTE_MAX_SOLVE_PIXELS must be 0 (unbounded) or a positive pixel count, got {raw!r}"
        ) from exc
    if value < 0:
        raise ValueError(
            f"MATTE_MAX_SOLVE_PIXELS must be 0 (unbounded) or a positive pixel count, got {value}"
        )
    return value or None


def solve_settings() -> tuple[str, int | None]:
    """(mode, pixel budget) from the environment. The budget only exists in tiled mode."""
    mode = solve_mode()
    return mode, (max_solve_pixels() if mode == TILED else None)


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


def solve_window(trimap: np.ndarray, margin: int = SOLVE_MARGIN_PX) -> tuple[int, int, int, int] | None:
    """Bounding box (y0, y1, x0, x1) of the pixels matting can affect, plus a margin.

    Alpha is fixed by the trimap outside the unknown band, so only the unknown and FG
    pixels (with surrounding context for the solver) need to enter the linear system.
    Returns None when the trimap is entirely known background.
    """
    rows = np.flatnonzero(trimap.any(axis=1))
    if rows.size == 0:
        return None
    cols = np.flatnonzero(trimap.any(axis=0))
    h, w = trimap.shape[:2]
    y0 = int(max(0, rows[0] - margin))
    y1 = int(min(h, rows[-1] + 1 + margin))
    x0 = int(max(0, cols[0] - margin))
    x1 = int(min(w, cols[-1] + 1 + margin))
    return y0, y1, x0, x1


def _closed_form(roi: np.ndarray, trimap: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    img = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB).astype(np.float64) / 255.0
    alpha = estimate_alpha_cf(img, trimap.astype(np.float64) / 255.0)
    fg = estimate_foreground_ml(img, alpha)
    fg_bgr = cv2.cvtColor((fg * 255).astype(np.uint8), cv2.COLOR_RGB2BGR).astype(np.float64) / 255.0
    return alpha, fg_bgr


Rect = tuple[int, int, int, int]


def tiles(h: int, w: int, budget: int | None) -> list[Rect]:
    """Cut h x w into rectangles small enough to be solved with context inside `budget`.

    Downscaling the solve instead would smear the hair-thin alpha detail the extraction
    exists for, so the window is cut into square-ish pieces solved at full resolution.
    Each piece is solved with surrounding context (`_label_aware_context`) and only the
    piece itself is written, so a tile edge does not become a seam in the alpha.
    """
    if budget is None or h * w <= budget:
        return [(0, h, 0, w)]
    side = max(1, int(budget**0.5))
    step = max(1, side - 2 * min(TILE_OVERLAP_PX, side // 4))
    return [(y, min(h, y + step), x, min(w, x + step)) for y in range(0, h, step) for x in range(0, w, step)]


def _holds_both_labels(trimap: np.ndarray, rect: Rect) -> bool:
    y0, y1, x0, x1 = rect
    context = trimap[y0:y1, x0:x1]
    return bool((context == 255).any() and (context == 0).any())


def _within(rect: Rect, budget: int | None) -> bool:
    y0, y1, x0, x1 = rect
    return budget is None or (y1 - y0) * (x1 - x0) <= budget


def _grown_to_hold_both_labels(trimap: np.ndarray, solved: Rect, budget: int | None) -> Rect | None:
    """Widen a solve rectangle until it contains known FG and BG, as the solver needs.

    A tile can easily hold nothing but unknown pixels (a wide unknown band, or a large
    user brush stroke). Answering 0 or 1 there would paint a rectangular block of hard
    alpha over pixels the solver can resolve from labels just outside the tile, so the
    context grows instead - but never past the budget, which is the whole reason the mode
    exists. Returns None when no budget-sized square context around the tile has both
    labels; `_label_aware_context` then reshapes the context instead of growing it.
    """
    y0, y1, x0, x1 = solved
    h, w = trimap.shape[:2]
    step = max(TILE_OVERLAP_PX, 1)
    while True:
        if _holds_both_labels(trimap, (y0, y1, x0, x1)):
            return y0, y1, x0, x1
        grown = (max(0, y0 - step), min(h, y1 + step), max(0, x0 - step), min(w, x1 + step))
        if grown == (y0, y1, x0, x1) or not _within(grown, budget):
            return None
        y0, y1, x0, x1 = grown


def _padded(written: Rect, h: int, w: int, budget: int | None) -> Rect:
    """`written` with the widest ring of context that still fits the budget.

    The ring is what keeps a tile edge from becoming a seam in the alpha: the solver reads
    it, only the tile itself is written back.
    """
    for pad in range(TILE_OVERLAP_PX, 0, -1):
        candidate = (
            max(0, written[0] - pad),
            min(h, written[1] + pad),
            max(0, written[2] - pad),
            min(w, written[3] + pad),
        )
        if _within(candidate, budget):
            return candidate
    return written


def _centred(span: tuple[int, int], length: int, size: int) -> tuple[int, int]:
    """A span of `length` pixels centred on `span`, clamped inside [0, size]."""
    start = min(max(0, (span[0] + span[1] - length) // 2), size - length)
    return start, start + length


def _label_aware_context(trimap: np.ndarray, written: Rect, budget: int | None) -> Rect | None:
    """Smallest rectangle around `written` that holds both labels and fits the budget.

    Tried in order of how local - and so how close to a full solve - the result is: a
    square-ish context grown outwards, then the tallest column and the widest row through
    the tile that the budget pays for. The strips reach labels sitting at opposite ends of
    the window (a wide unknown band with FG far below and BG far above), where a square
    context holding both would be many times the budget. Returns None when the budget buys
    no such rectangle; the caller then splits the tile and tries again.
    """
    h, w = trimap.shape[:2]
    square = _grown_to_hold_both_labels(trimap, _padded(written, h, w, budget), budget)
    if square is not None:
        return square
    limit = budget or h * w
    rows, cols = (written[0], written[1]), (written[2], written[3])
    tile_h, tile_w = rows[1] - rows[0], cols[1] - cols[0]
    column_w = max(tile_w, min(w, limit // h))
    row_h = max(tile_h, min(h, limit // w))
    for candidate in (
        (
            *_centred(rows, max(tile_h, min(h, limit // column_w)), h),
            *_centred(cols, column_w, w),
        ),
        (
            *_centred(rows, row_h, h),
            *_centred(cols, max(tile_w, min(w, limit // row_h)), w),
        ),
    ):
        if _within(candidate, budget) and _holds_both_labels(trimap, candidate):
            return candidate
    return None


def _bounded_tiles(trimap: np.ndarray, written: Rect, budget: int | None) -> list[tuple[Rect, Rect]]:
    """(written, solved) pairs covering `written`, every solve inside the budget.

    When no budget-sized context around a tile holds both labels, the tile is halved and
    each half retried: a narrower tile allows a taller context for the same pixel count,
    which is how a distant label is reached without exceeding the budget. Solving a
    label-less tile as hard 0/1 instead would paint a rectangular block of flat alpha.
    """
    y0, y1, x0, x1 = written
    if not (trimap[y0:y1, x0:x1] == 128).any():
        return []
    context = _label_aware_context(trimap, written, budget)
    if context is not None:
        return [(written, context)]
    if y1 - y0 <= 1 and x1 - x0 <= 1:
        raise ValueError(
            "MATTE_MAX_SOLVE_PIXELS is too small to solve this trimap: no solve region "
            f"within {budget} pixels contains both foreground and background labels. "
            "Raise the budget (or set MATTE_SOLVE_MODE=full) to matte this image."
        )
    if y1 - y0 >= x1 - x0:
        mid = (y0 + y1) // 2
        halves = ((y0, mid, x0, x1), (mid, y1, x0, x1))
    else:
        mid = (x0 + x1) // 2
        halves = ((y0, y1, x0, mid), (y0, y1, mid, x1))
    return [pair for half in halves for pair in _bounded_tiles(trimap, half, budget)]


def _solve_tiles(
    roi: np.ndarray,
    trimap: np.ndarray,
    alpha: np.ndarray,
    fg_bgr: np.ndarray,
    budget: int | None,
) -> list[int]:
    """Fill `alpha` / `fg_bgr` in place, one bounded tile of the window at a time.

    Returns the area of every solve it ran, which is what the per-run log reports: a solve
    larger than the budget is the regression this mode has to be watched for.
    """
    solved: list[int] = []
    h, w = trimap.shape[:2]
    for written in tiles(h, w, budget):
        for (y0, y1, x0, x1), (cy0, cy1, cx0, cx1) in _bounded_tiles(trimap, written, budget):
            tile_alpha, tile_fg = _closed_form(
                np.ascontiguousarray(roi[cy0:cy1, cx0:cx1]),
                np.ascontiguousarray(trimap[cy0:cy1, cx0:cx1]),
            )
            solved.append((cy1 - cy0) * (cx1 - cx0))
            sy, sx = y0 - cy0, x0 - cx0
            alpha[y0:y1, x0:x1] = tile_alpha[sy : sy + (y1 - y0), sx : sx + (x1 - x0)]
            fg_bgr[y0:y1, x0:x1] = tile_fg[sy : sy + (y1 - y0), sx : sx + (x1 - x0)]
    return solved


def _is_solvable(trimap: np.ndarray) -> bool:
    """Closed-form matting needs something unknown and both labels to propagate from."""
    return bool((trimap == 128).any() and (trimap == 255).any() and (trimap == 0).any())


def run_matting(
    roi_a: np.ndarray,
    trimap: np.ndarray,
    mode: str = FULL,
    max_solve_pixels: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Closed-form matting + ML foreground estimation. Returns (alpha float, fg float BGR).

    `full` (the default) solves the whole ROI in one system, which is what this repository
    has always done. `tiled` is the low-memory approximation: it solves only the window
    around the product (see `solve_window`), in pieces of at most `max_solve_pixels`
    pixels - a hard bound on every solve, context included (0 or None = the whole window
    at once). Outside the window alpha comes from the trimap and the foreground stays the
    source pixel. A budget too small to ever reach both labels raises rather than falling
    back to a whole-window solve, which is the OOM the mode exists to avoid.

    Degenerate trimaps - nothing unknown, or missing a label to propagate from - are
    answered from the trimap in both modes; pymatting raises on them.

    Each run logs the mode it used and the size of the largest solve, so an operator can
    confirm per click which path a deployment took.
    """
    if mode not in SOLVE_MODES:
        raise ValueError(f"solve mode must be one of {SOLVE_MODES}, got {mode!r}")
    started = time.perf_counter()
    solved: list[int] = []
    h, w = trimap.shape[:2]
    try:
        if not _is_solvable(trimap):
            return np.where(trimap == 255, 1.0, 0.0), roi_a.astype(np.float64) / 255.0
        if mode == FULL:
            solved = [h * w]
            return _closed_form(roi_a, trimap)
        alpha = np.where(trimap == 255, 1.0, 0.0)
        fg_bgr = roi_a.astype(np.float64) / 255.0
        y0, y1, x0, x1 = solve_window(trimap)
        solved = _solve_tiles(
            roi_a[y0:y1, x0:x1],
            trimap[y0:y1, x0:x1],
            alpha[y0:y1, x0:x1],
            fg_bgr[y0:y1, x0:x1],
            max_solve_pixels or None,
        )
        return alpha, fg_bgr
    finally:
        logger.info(
            "matte run: solve_mode=%s roi=%dx%d max_solve_pixels=%s solves=%d max_solve_px=%d elapsed_ms=%d",
            mode,
            w,
            h,
            max_solve_pixels or "none",
            len(solved),
            max(solved, default=0),
            (time.perf_counter() - started) * 1000,
        )


def composite(alpha: np.ndarray, fg_bgr: np.ndarray, base_bgr: np.ndarray) -> np.ndarray:
    a = alpha[..., None]
    out = a * fg_bgr + (1.0 - a) * (base_bgr.astype(np.float64) / 255.0)
    return (np.clip(out, 0, 1) * 255).astype(np.uint8)


BLEND_ROWS_PER_CHUNK = 64


def blend_rgba_over(
    rgba: np.ndarray, base_bgr: np.ndarray, rows_per_chunk: int = BLEND_ROWS_PER_CHUNK
) -> np.ndarray:
    """Alpha blend an RGBA layer over `base_bgr` in horizontal stripes.

    The arithmetic and the float64 precision are the ones this repository has always
    used, so the output is identical pixel for pixel; only the float64 buffers shrink
    from three whole frames (24 bytes per pixel each, +120..160MB on a phone photo) to a
    stripe at a time.
    """
    out = np.empty_like(base_bgr)
    for y0 in range(0, base_bgr.shape[0], rows_per_chunk):
        y1 = min(base_bgr.shape[0], y0 + rows_per_chunk)
        alpha = rgba[y0:y1, :, 3:4].astype(np.float64) / 255.0
        fg = rgba[y0:y1, :, :3].astype(np.float64) / 255.0
        base = base_bgr[y0:y1].astype(np.float64) / 255.0
        out[y0:y1] = (np.clip(alpha * fg + (1.0 - alpha) * base, 0, 1) * 255).astype(np.uint8)
    return out


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
    return blend_rgba_over(warped, edited_bgr)


def reconstruction_error(alpha: np.ndarray, fg_bgr: np.ndarray, roi_a: np.ndarray) -> float:
    """Mean abs error (0-255) inside alpha>0.05 when compositing back onto A itself."""
    recon = composite(alpha, fg_bgr, roi_a).astype(np.float32)
    mask = alpha > 0.05
    if not mask.any():
        return 0.0
    diff = np.abs(recon - roi_a.astype(np.float32))[mask]
    return float(diff.mean())
