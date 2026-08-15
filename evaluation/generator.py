"""Builds synthetic worn images whose ground truth is true by construction.

A case is `worn = alpha * product + (1 - alpha) * bare`, so the product's position,
shape and alpha are known exactly — no human annotation, and no annotator error.

Two deliberate choices keep the resulting numbers honest:

* the bare image is *not* a pixel-perfect copy of the worn background. It gets its own
  noise seed and, optionally, a small misalignment, because a real bare photo is a
  second shot. Without that, difference-based extraction is handed the exact equation
  used to build the data.
* the product's cast shadow and the person's own lashes are in the image but not in
  the ground truth mask; own lashes are marked in `gt_ignore` so the runner can report
  scores with and without them.
"""

from __future__ import annotations

import itertools
import json
import os
import random
import zlib
from dataclasses import dataclass, field
from typing import Any, Protocol

import cv2
import numpy as np

from evaluation.degrade import Degradation, degrade, misalign
from evaluation.synth import EyeBackground, Placement

ALPHA_MASK_THRESHOLD = 128  # gt_mask = gt_alpha >= 128 (i.e. alpha >= 0.5)


class Product(Protocol):
    name: str

    def render(self, background: EyeBackground, placement: Placement) -> tuple[np.ndarray, np.ndarray]: ...


@dataclass(frozen=True)
class CaseSpec:
    """Everything that makes one case, and nothing that depends on the pipeline."""

    case_id: str
    background: str
    product: str
    condition: str = "baseline"
    condition_value: Any = None
    block: int = 0
    placement: Placement = field(default_factory=Placement)
    worn: Degradation = field(default_factory=Degradation)
    bare: Degradation = field(default_factory=Degradation)
    bare_misalign_px: float = 0.0
    bare_misalign_deg: float = 0.0
    shadow: float = 0.0

    @property
    def seed(self) -> int:
        """Stable per-case seed, so regenerating a dataset reproduces it bit for bit.

        `hash()` is salted per process, so it must not be used here.
        """
        key = f"{self.case_id}/{self.background}/{self.product}".encode()
        return zlib.crc32(key)

    @property
    def pair_key(self) -> str:
        """Background + product. Every condition is compared against the baseline of
        the same pair, so a score change cannot come from a different eye."""
        return f"{self.background}|{self.product}"

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.case_id,
            "background": self.background,
            "product": self.product,
            "condition": self.condition,
            "condition_value": self.condition_value,
            "pair_key": self.pair_key,
            "block": int(self.block),
            "scale": float(self.placement.scale),
            "rotation_deg": float(self.placement.rotation_deg),
            "offset_x": float(self.placement.offset_x),
            "offset_y": float(self.placement.offset_y),
            "flip": bool(self.placement.flip),
            "shadow": float(self.shadow),
            "bare_misalign_px": float(self.bare_misalign_px),
            "bare_misalign_deg": float(self.bare_misalign_deg),
            **self.worn.as_dict(),
            "bare_degradation": self.bare.as_dict(),
        }


@dataclass(frozen=True)
class GeneratedCase:
    spec: CaseSpec
    bare: np.ndarray  # BGR uint8, the "without product" shot
    worn: np.ndarray  # BGR uint8, the "with product" shot
    gt_alpha: np.ndarray  # uint8 0..255
    gt_mask: np.ndarray  # uint8 0/255
    gt_ignore: np.ndarray  # bool: own lashes, excluded from scoring on request
    gt_product: np.ndarray  # BGRA uint8, the product as it was composited
    roi_rect: tuple[int, int, int, int] | None
    landmarks: np.ndarray | None


def _cast_shadow(alpha: np.ndarray, strength: float) -> np.ndarray:
    """Soft darkening under the lashes; present in the photo, absent from ground truth."""
    if strength <= 0:
        return np.zeros_like(alpha)
    height = alpha.shape[0]
    offset = max(1, int(round(height * 0.012)))
    shifted = np.zeros_like(alpha)
    shifted[offset:, :] = alpha[:-offset, :]
    return np.clip(cv2.GaussianBlur(shifted, (0, 0), max(1.0, height * 0.012)) * strength, 0.0, 1.0)


def build_case(background: EyeBackground, product: Product, spec: CaseSpec) -> GeneratedCase:
    product_bgr, alpha = product.render(background, spec.placement)
    base = background.image.astype(np.float32)

    shadow = _cast_shadow(alpha, spec.shadow)
    lit = base * (1.0 - 0.55 * shadow)[..., None]
    composited = product_bgr.astype(np.float32) * alpha[..., None] + lit * (1.0 - alpha[..., None])

    worn = degrade(np.clip(composited, 0, 255).astype(np.uint8), spec.worn, seed=spec.seed)
    bare = misalign(background.image, spec.bare_misalign_px, spec.bare_misalign_deg, seed=spec.seed + 1)
    # a different seed: two photos never share a noise field
    bare = degrade(bare, spec.bare, seed=spec.seed + 5000)

    gt_alpha = np.clip(np.rint(alpha * 255.0), 0, 255).astype(np.uint8)
    gt_mask = np.where(gt_alpha >= ALPHA_MASK_THRESHOLD, 255, 0).astype(np.uint8)
    own = background.own_lash_alpha
    gt_ignore = (own > 0.05) & (gt_alpha < ALPHA_MASK_THRESHOLD)
    return GeneratedCase(
        spec=spec,
        bare=bare,
        worn=worn,
        gt_alpha=gt_alpha,
        gt_mask=gt_mask,
        gt_ignore=gt_ignore,
        gt_product=np.dstack([product_bgr, gt_alpha]),
        roi_rect=background.roi_rect,
        landmarks=background.landmarks,
    )


MANIFEST = "manifest.json"


def write_manifest(root: str, specs: list[CaseSpec], extra: dict[str, Any] | None = None) -> str:
    """Record exactly which cases belong to this dataset.

    Without it, regenerating a smaller dataset into an existing directory leaves the
    older, larger set of case folders behind and the runner silently evaluates both.
    """
    path = os.path.join(root, MANIFEST)
    payload = {
        "cases": [spec.case_id for spec in specs],
        "blocks": complete_blocks(specs),
        "block_size": BLOCK_SIZE,
        **(extra or {}),
    }
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    return path


def write_case(root: str, case: GeneratedCase) -> str:
    directory = os.path.join(root, case.spec.case_id)
    os.makedirs(directory, exist_ok=True)
    cv2.imwrite(os.path.join(directory, "bare.png"), case.bare)
    cv2.imwrite(os.path.join(directory, "worn.png"), case.worn)
    cv2.imwrite(os.path.join(directory, "gt_alpha.png"), case.gt_alpha)
    cv2.imwrite(os.path.join(directory, "gt_mask.png"), case.gt_mask)
    cv2.imwrite(os.path.join(directory, "gt_ignore.png"), case.gt_ignore.astype(np.uint8) * 255)
    cv2.imwrite(os.path.join(directory, "gt_product.png"), case.gt_product)
    metadata = {
        **case.spec.as_dict(),
        "width": int(case.worn.shape[1]),
        "height": int(case.worn.shape[0]),
        "roi_rect": list(case.roi_rect) if case.roi_rect is not None else None,
        "has_landmarks": case.landmarks is not None,
        "own_lash_ignore_available": bool(case.gt_ignore.any()),
    }
    with open(os.path.join(directory, "metadata.json"), "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2)
    if case.landmarks is not None:
        np.save(os.path.join(directory, "landmarks.npy"), case.landmarks)
    return directory


# One axis at a time around a pristine baseline: a score drop can then be attributed
# to a single condition instead of an unknown mixture of them.
AXES: tuple[tuple[str, tuple[Any, ...]], ...] = (
    ("scale", (0.85, 1.15)),
    ("rotation_deg", (-10.0, -5.0, 5.0, 10.0)),
    ("offset", ((3.0, -2.0), (-5.0, 4.0))),
    ("flip", (True,)),
    ("brightness", (0.7, 1.3)),
    ("contrast", (0.75,)),
    ("blur_sigma", (1.0, 2.0)),
    ("noise_sigma", (3.0, 8.0)),
    ("jpeg_quality", (60, 30)),
    ("bare_misalign_px", (1.5, 4.0)),
    ("shadow", (0.35,)),
    # the bare shot differs from the worn shot in exposure/focus, not just in noise.
    # This is the axis that actually stresses difference-based extraction.
    ("exposure_mismatch", (0.9, 1.15)),
    ("bare_blur_mismatch", (1.2,)),
    ("realistic", ("noise+jpeg+misalign+shadow",)),
)


def _apply_axis(spec_kwargs: dict[str, Any], axis: str, value: Any) -> None:
    if axis == "scale":
        spec_kwargs["placement"] = Placement(scale=value)
    elif axis == "rotation_deg":
        spec_kwargs["placement"] = Placement(rotation_deg=value)
    elif axis == "offset":
        spec_kwargs["placement"] = Placement(offset_x=value[0], offset_y=value[1])
    elif axis == "flip":
        spec_kwargs["placement"] = Placement(flip=value)
    elif axis in {"brightness", "contrast", "blur_sigma", "noise_sigma", "jpeg_quality"}:
        # the worn photo changes; the bare photo shares the exposure but not the noise
        spec_kwargs["worn"] = Degradation(**{axis: value})
        spec_kwargs["bare"] = Degradation(**{axis: value})
    elif axis == "bare_misalign_px":
        spec_kwargs["bare_misalign_px"] = value
        spec_kwargs["bare_misalign_deg"] = value / 4.0
    elif axis == "shadow":
        spec_kwargs["shadow"] = value
    elif axis == "exposure_mismatch":
        # only the bare shot changes: the two photos were taken under different light
        spec_kwargs["bare"] = Degradation(brightness=value)
    elif axis == "bare_blur_mismatch":
        spec_kwargs["bare"] = Degradation(blur_sigma=value)
    elif axis == "realistic":
        spec_kwargs["worn"] = Degradation(noise_sigma=3.0, jpeg_quality=75)
        spec_kwargs["bare"] = Degradation(noise_sigma=3.0, jpeg_quality=75)
        spec_kwargs["bare_misalign_px"] = 1.5
        spec_kwargs["bare_misalign_deg"] = 0.4
        spec_kwargs["shadow"] = 0.3
    else:
        raise ValueError(f"unknown axis: {axis}")


def variants_of(axes: tuple[tuple[str, tuple[Any, ...]], ...] = AXES) -> list[tuple[str, Any]]:
    """The baseline plus every axis value, in a fixed order. One block of cases."""
    return [("baseline", None)] + [(axis, value) for axis, values in axes for value in values]


BLOCK_SIZE = len(variants_of())


def plan_cases(
    backgrounds: list[str],
    products: list[str],
    count: int,
    seed: int = 0,
    axes: tuple[tuple[str, tuple[Any, ...]], ...] = AXES,
) -> list[CaseSpec]:
    """Enumerate `count` case specs as *blocks*: one background+product pair per block,
    holding the baseline and every axis value of that same pair.

    Blocking is what makes the condition breakdown meaningful. Cycling conditions and
    backgrounds independently would compare, say, the JPEG cases of one set of eyes
    against the baseline of a different set, so the difference would partly measure the
    eye rather than the condition.

    `seed` permutes the order of the pairs, which decides who gets included when `count`
    does not cover every pair.
    """
    if not backgrounds or not products:
        raise ValueError("need at least one background and one product")
    variants = variants_of(axes)
    pairs = [(background, product) for product in products for background in backgrounds]
    random.Random(seed).shuffle(pairs)

    specs: list[CaseSpec] = []
    for block, (background, product) in enumerate(itertools.cycle(pairs)):
        if len(specs) >= count:
            break
        for axis, value in variants:
            if len(specs) >= count:
                break
            kwargs: dict[str, Any] = {
                "case_id": f"case_{len(specs) + 1:04d}",
                "background": background,
                "product": product,
                "condition": axis,
                "condition_value": value,
                "block": block,
            }
            if axis != "baseline":
                _apply_axis(kwargs, axis, value)
            specs.append(CaseSpec(**kwargs))
    return specs


def complete_blocks(specs: list[CaseSpec]) -> int:
    """How many pairs got every variant. Only these support a paired comparison."""
    counted: dict[int, int] = {}
    for spec in specs:
        counted[spec.block] = counted.get(spec.block, 0) + 1
    return sum(1 for total in counted.values() if total == BLOCK_SIZE)
