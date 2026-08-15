"""The two kinds of product a case can be built from.

`ProceduralProduct` is the default because its ground truth alpha is exact at every
scale and rotation: only the strand geometry is transformed, and rasterisation happens
once. `ImageProduct` accepts a real product cut-out (RGBA PNG) so results can be
sanity-checked against the actual article, at the cost of one resampling pass in the
ground truth itself — reported honestly rather than hidden.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from evaluation.synth import (
    EyeBackground,
    LashGeometry,
    Placement,
    placement_matrix,
    render_geometry,
    synthesize_product,
)


@dataclass(frozen=True)
class ProceduralProduct:
    seed: int = 0
    n_strands: int = 40
    length: float = 0.34
    curl: float = 0.45
    thickness: float = 1.6

    @property
    def name(self) -> str:
        return f"procedural_lash_{self.seed:04d}"

    @property
    def exact_ground_truth(self) -> bool:
        return True

    def geometry(self, background: EyeBackground) -> LashGeometry:
        return synthesize_product(
            background,
            seed=self.seed,
            n_strands=self.n_strands,
            length=self.length,
            curl=self.curl,
            thickness=self.thickness,
        )

    def render(self, background: EyeBackground, placement: Placement) -> tuple[np.ndarray, np.ndarray]:
        geometry = self.geometry(background)
        placed = geometry.transformed(placement_matrix(placement, background.centre))
        return render_geometry(placed, background.shape)


def load_product_png(path: str) -> np.ndarray:
    rgba = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if rgba is None:
        raise ValueError(f"could not read product image: {path}")
    if rgba.ndim != 3 or rgba.shape[2] != 4:
        raise ValueError(f"product image must be RGBA with an alpha channel: {path}")
    return rgba


@dataclass(frozen=True)
class ImageProduct:
    """A real cut-out product photo, fitted onto the lash line of a background."""

    rgba: np.ndarray
    name: str = "product_png"
    width_ratio: float = 1.05  # product width relative to the lash line span

    @property
    def exact_ground_truth(self) -> bool:
        return False

    def _fit_matrix(self, background: EyeBackground) -> np.ndarray:
        line = background.lash_line
        span = float(np.linalg.norm(line[-1] - line[0]))
        alpha = self.rgba[:, :, 3]
        ys, xs = np.nonzero(alpha > 8)
        if len(xs) == 0:
            raise ValueError("product image has an empty alpha channel")
        x0, x1, y1 = float(xs.min()), float(xs.max() + 1), float(ys.max() + 1)
        scale = span * self.width_ratio / max(1.0, x1 - x0)
        anchor = line[len(line) // 2]
        matrix = np.array([[scale, 0.0, 0.0], [0.0, scale, 0.0], [0.0, 0.0, 1.0]])
        # put the bottom centre of the product on the middle of the lash line
        matrix[0, 2] = anchor[0] - (x0 + x1) / 2 * scale
        matrix[1, 2] = anchor[1] - y1 * scale
        return matrix

    def render(self, background: EyeBackground, placement: Placement) -> tuple[np.ndarray, np.ndarray]:
        total = np.vstack([placement_matrix(placement, background.centre), [0, 0, 1]]) @ self._fit_matrix(
            background
        )
        height, width = background.shape
        alpha = self.rgba[:, :, 3].astype(np.float32) / 255.0
        premultiplied = np.dstack([self.rgba[:, :, :3].astype(np.float32) * alpha[..., None], alpha])
        scale = float(np.sqrt(abs(np.linalg.det(total[:2, :2]))))
        # warpAffine has no INTER_AREA, so a shrink is the one case where the ground
        # truth of an image product is slightly soft. Procedural products avoid this.
        flags = cv2.INTER_LINEAR if scale < 1 else cv2.INTER_LANCZOS4
        warped = cv2.warpAffine(
            premultiplied,
            total[:2],
            (width, height),
            flags=flags,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        out_alpha = np.clip(warped[:, :, 3], 0.0, 1.0)
        safe = np.where(out_alpha > 1e-6, out_alpha, 1.0)[..., None]
        bgr = np.clip(warped[:, :, :3] / safe, 0, 255).astype(np.uint8)
        return bgr, out_alpha
