"""Product asset domain values."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class AssetDraft:
    """An extracted lash ready to be catalogued."""

    name: str
    rgba: np.ndarray
    roi: tuple[int, int, int, int]
    roi_scale: float
    landmarks: np.ndarray
    brand: str | None = None
    session_id: str | None = None
    recon_error: float | None = None

    def validate(self) -> None:
        if not self.name.strip():
            raise ValueError("asset name is required")
        if self.rgba.ndim != 3 or self.rgba.shape[2] != 4:
            raise ValueError("product image must be RGBA")
        if len(self.roi) != 4:
            raise ValueError("roi must be [x0, y0, x1, y1]")

    @property
    def alpha(self) -> np.ndarray:
        return self.rgba[..., 3].astype(np.float32) / 255.0


@dataclass(frozen=True)
class ProductAsset:
    id: str
    name: str
    storage_path: str
    brand: str | None
    session_id: str | None
    width: int
    height: int
    roi: list[int]
    roi_scale: float
    alpha_coverage: float | None
    recon_error: float | None

    @classmethod
    def from_row(cls, row: dict[str, Any]) -> ProductAsset:
        return cls(
            id=row["id"],
            name=row["name"],
            storage_path=row["storage_path"],
            brand=row.get("brand"),
            session_id=row.get("session_id"),
            width=row["width"],
            height=row["height"],
            roi=list(row["roi"]),
            roi_scale=row["roi_scale"],
            alpha_coverage=row.get("alpha_coverage"),
            recon_error=row.get("recon_error"),
        )
