"""Capture conditions: exposure, focus, sensor noise, JPEG, and camera movement.

These are not decoration. With a pixel-perfect bare image, `worn - bare` is exactly
`alpha * (product - bare)`, so difference-based extraction becomes the very equation
the synthetic data was built from and every score saturates. Real pairs are two
different photos: independent sensor noise, slightly different exposure, and a head
that moved. Noise is therefore drawn from a per-image seed, and `misalign` is applied
to the bare image only.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class Degradation:
    brightness: float = 1.0
    contrast: float = 1.0
    blur_sigma: float = 0.0
    noise_sigma: float = 0.0
    jpeg_quality: int = 0  # 0 = keep it lossless

    @property
    def is_identity(self) -> bool:
        return (
            self.brightness == 1.0
            and self.contrast == 1.0
            and self.blur_sigma == 0.0
            and self.noise_sigma == 0.0
            and self.jpeg_quality == 0
        )

    def as_dict(self) -> dict[str, float]:
        return {
            "brightness": float(self.brightness),
            "contrast": float(self.contrast),
            "blur_sigma": float(self.blur_sigma),
            "noise_sigma": float(self.noise_sigma),
            "jpeg_quality": int(self.jpeg_quality),
        }


def degrade(image: np.ndarray, degradation: Degradation, seed: int) -> np.ndarray:
    """Apply exposure -> focus -> sensor noise -> JPEG, in capture order."""
    if degradation.is_identity:
        return image.copy()
    out = image.astype(np.float32)
    if degradation.contrast != 1.0:
        out = (out - out.mean()) * degradation.contrast + out.mean()
    if degradation.brightness != 1.0:
        out *= degradation.brightness
    if degradation.blur_sigma > 0:
        out = cv2.GaussianBlur(out, (0, 0), degradation.blur_sigma)
    if degradation.noise_sigma > 0:
        rng = np.random.default_rng(seed)
        out += rng.normal(0.0, degradation.noise_sigma, out.shape).astype(np.float32)
    out = np.clip(out, 0, 255).astype(np.uint8)
    if degradation.jpeg_quality:
        ok, buffer = cv2.imencode(".jpg", out, [int(cv2.IMWRITE_JPEG_QUALITY), degradation.jpeg_quality])
        if not ok:
            raise RuntimeError("JPEG encoding failed")
        out = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    return out


def misalign(image: np.ndarray, shift_px: float, rotation_deg: float, seed: int) -> np.ndarray:
    """Move the image as if it were a second shot taken a moment later.

    Applied to the bare image, so the pipeline has to align it (landmark affine + ECC)
    exactly like it does for real photo pairs.
    """
    if shift_px == 0.0 and rotation_deg == 0.0:
        return image.copy()
    rng = np.random.default_rng(seed)
    angle = rng.uniform(0, 2 * np.pi)
    height, width = image.shape[:2]
    matrix = cv2.getRotationMatrix2D((width / 2, height / 2), rotation_deg, 1.0)
    matrix[0, 2] += np.cos(angle) * shift_px
    matrix[1, 2] += np.sin(angle) * shift_px
    return cv2.warpAffine(
        image, matrix, (width, height), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE
    )
