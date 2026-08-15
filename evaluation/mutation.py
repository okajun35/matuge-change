"""How much do product pixels change when they are moved?

"Pixel Preserve" is the core promise of the project, but every path that places the
extracted product somewhere else resamples it:

* `matting.recompose_onto` warps the RGBA with `INTER_LINEAR` **without premultiplying
  alpha**, so the colour stored in fully transparent pixels bleeds into the lash tips;
* `SessionService._recompose_into_rect` (manual ROI) premultiplies first, so the two
  paths do not behave the same way;
* `roi.crop_roi` shrinks any ROI wider than `MAX_ROI_WIDTH` with `INTER_AREA` before
  extraction even starts, which is what happens to every full-resolution phone photo.

This module measures those effects. It never changes production code: `warp_product`
with `interpolation="linear", premultiply=False` is verified to reproduce
`recompose_onto` bit for bit (see tests/evaluation/test_mutation.py), and the other
variants exist only for comparison.
"""

from __future__ import annotations

import cv2
import numpy as np

from backend.lash_extraction import ALIGN_POINTS, MAX_ROI_WIDTH, EyeRoi
from evaluation import metrics
from evaluation.synth import EyeBackground, Placement, placement_matrix

INTERPOLATIONS = {
    "nearest": cv2.INTER_NEAREST,
    "linear": cv2.INTER_LINEAR,
    "lanczos4": cv2.INTER_LANCZOS4,
}

# (interpolation, premultiply) pairs to compare. "linear" is what production does today.
VARIANTS: dict[str, tuple[str, bool]] = {
    "nearest": ("nearest", False),
    "linear": ("linear", False),
    "premultiplied_linear": ("linear", True),
    "premultiplied_lanczos4": ("lanczos4", True),
}

TRANSFORMS: dict[str, Placement] = {
    "identity": Placement(),
    "translate_int": Placement(offset_x=3, offset_y=-2),
    "translate_fraction": Placement(offset_x=0.5, offset_y=0.5),
    "rotate_5": Placement(rotation_deg=5),
    "rotate_10": Placement(rotation_deg=10),
    "scale_down": Placement(scale=0.9),
    "scale_up": Placement(scale=1.1),
}


def transform_matrix(name: str, shape: tuple[int, int]) -> np.ndarray:
    if name not in TRANSFORMS:
        raise ValueError(f"unknown transform: {name}")
    height, width = shape[:2]
    return placement_matrix(TRANSFORMS[name], (width / 2, height / 2))


def warp_product(
    rgba: np.ndarray,
    matrix: np.ndarray,
    size: tuple[int, int],
    interpolation: str = "linear",
    premultiply: bool = False,
) -> np.ndarray:
    """Warp a BGRA product. `premultiply=False` is what `recompose_onto` does."""
    flags = INTERPOLATIONS[interpolation]
    if not premultiply:
        return cv2.warpAffine(
            rgba,
            matrix,
            size,
            flags=flags,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0, 0),
        )
    alpha = rgba[:, :, 3].astype(np.float32) / 255.0
    source = np.dstack([rgba[:, :, :3].astype(np.float32) * alpha[..., None], alpha])
    warped = cv2.warpAffine(source, matrix, size, flags=flags, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    out_alpha = np.clip(warped[:, :, 3], 0.0, 1.0)
    safe = np.where(out_alpha > 1e-6, out_alpha, 1.0)[..., None]
    colour = np.clip(warped[:, :, :3] / safe, 0, 255)
    return np.dstack([colour, out_alpha * 255.0]).round().astype(np.uint8)


def nearest_source_colors(rgba: np.ndarray, matrix: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """The colour each output pixel would have if the move were lossless (no blending)."""
    return cv2.warpAffine(
        rgba[:, :, :3],
        matrix,
        size,
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )


def composite_over(rgba: np.ndarray, base_bgr: np.ndarray) -> np.ndarray:
    """Alpha blend exactly as `recompose_onto` finishes its work."""
    alpha = rgba[..., 3:4].astype(np.float64) / 255.0
    foreground = rgba[..., :3].astype(np.float64) / 255.0
    base = base_bgr.astype(np.float64) / 255.0
    return (np.clip(alpha * foreground + (1.0 - alpha) * base, 0, 1) * 255).astype(np.uint8)


def recompose_matrix(roi: EyeRoi, lms_worn: np.ndarray, lms_edited: np.ndarray) -> np.ndarray:
    """The ROI -> edited-image affine that `recompose_onto` builds internally."""
    source = lms_worn[ALIGN_POINTS].astype(np.float32)
    destination = lms_edited[ALIGN_POINTS].astype(np.float32)
    affine, _ = cv2.estimateAffinePartial2D(source, destination, method=cv2.LMEDS)
    if affine is None:
        raise ValueError("could not estimate the landmark affine")
    inverse_scale = 1.0 / roi.scale
    roi_to_worn = np.array(
        [[inverse_scale, 0, roi.x0], [0, inverse_scale, roi.y0], [0, 0, 1]], dtype=np.float64
    )
    return (np.vstack([affine, [0, 0, 1]]) @ roi_to_worn)[:2]


def run_experiment(
    background: EyeBackground,
    product: object,
    transforms: tuple[str, ...] = tuple(TRANSFORMS),
    variants: dict[str, tuple[str, bool]] | None = None,
    opaque_min: int = 230,
    fill_transparent: bool = True,
) -> list[dict[str, object]]:
    """Compare pixel-warped products against the exactly rendered ground truth.

    The reference for each transform is the product *re-rendered from geometry* at that
    transform, so geometric distortion (`alpha_mad`, `alpha_grad`) and colour mutation
    are separated instead of being lumped into one number.

    `fill_transparent` puts the background's own colour into fully transparent pixels,
    so that a non-premultiplied warp has something to drag into the lash tips (a rendered
    product has black there, which would hide the effect entirely).

    Treat the resulting `fringe_rgb_mae` as an **upper bound, not a prediction**. The real
    `estimate_foreground_ml` does not leave skin colour there: measured on this pipeline it
    leaves luminance ~86 while the skin is ~178 and the lash body ~60, so the actual colour
    bleed is tiny (< 0.25/255 — see docs/benchmark-findings.md §5.1). Anything measured with
    this flag says "how bad could it get", not "how bad it is".
    """
    variants = variants or VARIANTS
    height, width = background.shape
    base_bgr, base_alpha = product.render(background, Placement())  # type: ignore[attr-defined]
    if fill_transparent:
        base_bgr = base_bgr.copy()
        empty = base_alpha < 0.03
        base_bgr[empty] = background.image[empty]
    base_rgba = np.dstack([base_bgr, np.rint(base_alpha * 255).astype(np.uint8)])

    rows: list[dict[str, object]] = []
    for name in transforms:
        matrix = transform_matrix(name, (height, width))
        reference_bgr, reference_alpha = product.render(background, TRANSFORMS[name])  # type: ignore[attr-defined]
        lossless = nearest_source_colors(base_rgba, matrix, (width, height))
        for variant, (interpolation, premultiply) in variants.items():
            warped = warp_product(base_rgba, matrix, (width, height), interpolation, premultiply)
            opaque = warped[:, :, 3] >= opaque_min
            colour = metrics.pixel_mutation(lossless, warped[:, :, :3], opaque)
            alpha_errors = metrics.matting_errors(warped[:, :, 3], reference_alpha)
            against_reference = metrics.product_fidelity(
                warped[:, :, :3], reference_bgr, warped[:, :, 3], reference_alpha
            )
            # the fringe is where a non-premultiplied warp drags background colour in
            warped_alpha = warped[:, :, 3].astype(np.float32) / 255.0
            fringe = (warped_alpha > 0.08) & (warped_alpha < 0.92) & (reference_alpha > 0.08)
            fringe_error = metrics.region_fidelity(warped[:, :, :3], reference_bgr, fringe)
            rows.append(
                {
                    "transform": name,
                    "variant": variant,
                    "interpolation": interpolation,
                    "premultiplied": premultiply,
                    "production_default": variant == "linear",
                    "alpha_mad": alpha_errors["mad"],
                    "alpha_grad": alpha_errors["grad"],
                    "rgb_mae_vs_reference": against_reference["rgb_mae"],
                    "fringe_rgb_mae": fringe_error["mae"],
                    **colour,
                }
            )
    return rows


def roi_downscale_experiment(
    image: np.ndarray,
    widths: tuple[int, ...] = (MAX_ROI_WIDTH, 800, 500, 300),
) -> list[dict[str, object]]:
    """What `crop_roi`'s INTER_AREA shrink costs, per target ROI width.

    Compared against an INTER_NEAREST resize to the same size, which by definition
    keeps original pixel values: any difference is interpolation, not scaling.
    """
    height, width = image.shape[:2]
    rows: list[dict[str, object]] = []
    for target in widths:
        scale = min(1.0, target / width)
        if scale >= 1.0:
            shrunk = image.copy()
            lossless = image.copy()
        else:
            size = (max(1, int(round(width * scale))), max(1, int(round(height * scale))))
            shrunk = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
            lossless = cv2.resize(image, size, interpolation=cv2.INTER_NEAREST)
        mask = np.ones(shrunk.shape[:2], bool)
        rows.append(
            {
                "roi_width": target,
                "scale": round(scale, 4),
                "interpolation": "none" if scale >= 1.0 else "INTER_AREA",
                **metrics.pixel_mutation(lossless, shrunk, mask),
            }
        )
    return rows
