"""Shape descriptor of a lash alpha matte, used for similarity search."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

DESCRIPTOR_DIM = 64
_BLOCK = DESCRIPTOR_DIM // 4
_NORMALISED_SIZE = 128
_VISIBLE = 0.05


def alpha_coverage(alpha: np.ndarray, thresh: float = _VISIBLE) -> float:
    """Fraction of the matte that is visibly opaque."""
    return float((alpha > thresh).mean())


@dataclass(frozen=True)
class LashDescriptor:
    """Scale- and translation-invariant descriptor of a lash shape.

    Four L2-normalised 16-bin blocks: horizontal density profile (spread along
    the lash line), vertical density profile (length distribution), alpha value
    histogram (softness) and gradient orientation histogram (curl direction).
    """

    values: np.ndarray

    @property
    def is_empty(self) -> bool:
        return not bool(self.values.any())

    def to_list(self) -> list[float]:
        return [float(v) for v in self.values]

    @classmethod
    def from_alpha(cls, alpha: np.ndarray) -> LashDescriptor:
        if alpha.ndim != 2:
            raise ValueError("alpha matte must be a 2-D array")
        a = np.clip(alpha.astype(np.float32), 0.0, 1.0)
        ys, xs = np.nonzero(a > _VISIBLE)
        if ys.size == 0:
            return cls(np.zeros(DESCRIPTOR_DIM, np.float32))

        a = a[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1]
        a = cv2.resize(a, (_NORMALISED_SIZE, _NORMALISED_SIZE), interpolation=cv2.INTER_AREA)

        columns = a.sum(axis=0).reshape(_BLOCK, -1).sum(axis=1)
        rows = a.sum(axis=1).reshape(_BLOCK, -1).sum(axis=1)
        opacity = np.histogram(a[a > 0.02], bins=_BLOCK, range=(0.0, 1.0))[0]

        gx = cv2.Sobel(a, cv2.CV_32F, 1, 0)
        gy = cv2.Sobel(a, cv2.CV_32F, 0, 1)
        angles = np.rad2deg(np.arctan2(gy, gx)) % 180.0
        orientation = np.histogram(angles, bins=_BLOCK, range=(0.0, 180.0), weights=cv2.magnitude(gx, gy))[0]

        blocks = [columns, rows, opacity, orientation]
        unit = [b.astype(np.float32) / (np.linalg.norm(b) + 1e-8) for b in blocks]
        return cls(np.concatenate(unit).astype(np.float32))
