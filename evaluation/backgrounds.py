"""Where bare (product-free) eye images come from.

Two sources, deliberately kept behind one type so the benchmark does not depend on
any dataset:

* `procedural_backgrounds` — synthesised eyes. No download, no licence, and the
  person's own lashes are known exactly, which is what lets the runner separate
  "found the product" from "found the model's own lashes".
* `image_backgrounds` — any folder of JPEG/PNG photos. MediaPipe locates the upper
  lids, so the product is placed on the lash line instead of in the middle of the
  frame. Photos are never committed to the repository.

MediaPipe cannot detect a face in an eye close-up or a profile (see docs/handover.md
§8), which is also true of the procedural eyes here: those cases are evaluated through
the pipeline's manual-ROI mode, with a *fixed* ROI rule that does not look at the
product, so no ground truth leaks into the input.
"""

from __future__ import annotations

import os
from collections.abc import Iterator

import cv2
import numpy as np

from backend.lash_extraction import LEFT_EYE, RIGHT_EYE, detect_landmarks
from evaluation.synth import EyeBackground, synthesize_bare_eye

IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".webp", ".bmp")

# MediaPipe upper-lid rings, ordered corner -> corner along the lash line.
UPPER_LID_RIGHT = [33, 246, 161, 160, 159, 158, 157, 173, 133]
UPPER_LID_LEFT = [362, 398, 384, 385, 386, 387, 388, 466, 263]


def fixed_roi_rect(eye_box: tuple[int, int, int, int], shape: tuple[int, int]) -> tuple[int, int, int, int]:
    """Manual ROI for faceless sources, using the same margins as `compute_eye_roi`.

    The rule depends only on the eye, never on where the product was placed, so
    rotated or scaled cases get exactly the same ROI as the baseline.
    """
    height, width = shape[:2]
    x0, y0, x1, y1 = eye_box
    eye_w, eye_h = x1 - x0, y1 - y0
    return (
        int(max(0, x0 - 0.45 * eye_w)),
        int(max(0, y0 - 2.4 * eye_h)),
        int(min(width, x1 + 0.45 * eye_w)),
        int(min(height, y1 + 1.4 * eye_h)),
    )


def _with_roi(background: EyeBackground) -> EyeBackground:
    from dataclasses import replace

    if background.landmarks is not None:
        return background  # a detectable face: let the pipeline find its own ROI
    return replace(background, roi_rect=fixed_roi_rect(background.eye_box, background.shape))


def procedural_backgrounds(
    count: int,
    width: int = 320,
    height: int = 240,
    seed: int = 0,
    own_lash: float = 1.0,
) -> list[EyeBackground]:
    return [
        _with_roi(synthesize_bare_eye(width, height, seed=seed + i, own_lash=own_lash)) for i in range(count)
    ]


def _eye_background_from_photo(image: np.ndarray, name: str) -> EyeBackground | None:
    landmarks = detect_landmarks(image)
    if landmarks is None:
        return None
    lines = tuple(np.asarray(landmarks[indices], np.float32) for indices in (UPPER_LID_RIGHT, UPPER_LID_LEFT))
    eye_points = landmarks[LEFT_EYE + RIGHT_EYE]
    height, width = image.shape[:2]
    eye_box = (
        int(max(0, eye_points[:, 0].min())),
        int(max(0, eye_points[:, 1].min())),
        int(min(width, eye_points[:, 0].max() + 1)),
        int(min(height, eye_points[:, 1].max() + 1)),
    )
    return EyeBackground(
        image=image,
        own_lash_alpha=np.zeros((height, width), np.float32),  # unknown for real photos
        lash_lines=lines,
        eye_box=eye_box,
        name=name,
        landmarks=landmarks,
    )


def image_backgrounds(
    directory: str,
    limit: int | None = None,
    max_width: int = 900,
    on_no_face: str = "skip",
) -> Iterator[EyeBackground]:
    """Backgrounds from a local folder of photos.

    `max_width` downscales large photos: the pipeline resamples any ROI wider than
    `MAX_ROI_WIDTH` with INTER_AREA, and that resampling would otherwise be mixed
    into the product fidelity numbers.
    `on_no_face` is `skip` (default) or `fallback` (place on a fixed rectangle and
    evaluate in manual-ROI mode).
    """
    if on_no_face not in {"skip", "fallback"}:
        raise ValueError("on_no_face must be 'skip' or 'fallback'")
    names = sorted(n for n in os.listdir(directory) if n.lower().endswith(IMAGE_SUFFIXES))
    produced = 0
    for name in names:
        if limit is not None and produced >= limit:
            return
        image = cv2.imread(os.path.join(directory, name), cv2.IMREAD_COLOR)
        if image is None:
            continue
        if image.shape[1] > max_width:
            scale = max_width / image.shape[1]
            image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        background = _eye_background_from_photo(image, name)
        if background is None:
            if on_no_face == "skip":
                continue
            background = _fallback_background(image, name)
        produced += 1
        yield _with_roi(background)


def _fallback_background(image: np.ndarray, name: str) -> EyeBackground:
    """No face detected: assume an eye close-up filling the middle of the frame."""
    height, width = image.shape[:2]
    x0, x1 = int(width * 0.15), int(width * 0.85)
    y0, y1 = int(height * 0.42), int(height * 0.62)
    line = np.stack(
        [
            np.linspace(x0, x1, 15, dtype=np.float32),
            y0 + (y1 - y0) * 0.5 - (y1 - y0) * 0.35 * np.sin(np.linspace(0, np.pi, 15, dtype=np.float32)),
        ],
        axis=1,
    ).astype(np.float32)
    return EyeBackground(
        image=image,
        own_lash_alpha=np.zeros((height, width), np.float32),
        lash_lines=(line,),
        eye_box=(x0, y0, x1, y1),
        name=name,
    )
