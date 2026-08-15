"""Reading a benchmark dataset back from disk.

A case is just a folder: `worn.png`, optional `bare.png`, `gt_alpha.png`, `gt_mask.png`,
optional `gt_ignore.png` / `gt_product.png`, and `metadata.json`. Nothing here is specific
to synthetic data, so a folder of real photographed cases (worn / bare / traced mask) can
be evaluated by the same runner once such a set exists.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np


@dataclass(frozen=True)
class Case:
    case_id: str
    directory: str
    metadata: dict[str, Any]

    def _read(self, name: str, flags: int) -> np.ndarray | None:
        path = os.path.join(self.directory, name)
        if not os.path.exists(path):
            return None
        return cv2.imread(path, flags)

    @property
    def worn(self) -> np.ndarray:
        image = self._read("worn.png", cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"{self.case_id}: worn.png is required")
        return image

    @property
    def bare(self) -> np.ndarray | None:
        return self._read("bare.png", cv2.IMREAD_COLOR)

    @property
    def gt_alpha(self) -> np.ndarray:
        alpha = self._read("gt_alpha.png", cv2.IMREAD_GRAYSCALE)
        if alpha is None:
            raise FileNotFoundError(f"{self.case_id}: gt_alpha.png is required")
        return alpha

    @property
    def gt_mask(self) -> np.ndarray:
        mask = self._read("gt_mask.png", cv2.IMREAD_GRAYSCALE)
        return (self.gt_alpha >= 128).astype(np.uint8) * 255 if mask is None else mask

    @property
    def gt_ignore(self) -> np.ndarray | None:
        ignore = self._read("gt_ignore.png", cv2.IMREAD_GRAYSCALE)
        return None if ignore is None else ignore > 0

    @property
    def gt_product(self) -> np.ndarray | None:
        return self._read("gt_product.png", cv2.IMREAD_UNCHANGED)

    @property
    def roi_rect(self) -> tuple[float, float, float, float] | None:
        rect = self.metadata.get("roi_rect")
        return None if rect is None else tuple(float(v) for v in rect)  # type: ignore[return-value]

    @property
    def condition(self) -> str:
        return str(self.metadata.get("condition", "unknown"))


MANIFEST = "manifest.json"


def load_dataset(root: str) -> list[Case]:
    """Load every case of a dataset.

    If a `manifest.json` is present, only the cases it lists are loaded. Regenerating a
    smaller dataset over an older one leaves the older case folders on disk, and without
    the manifest they would be silently evaluated as part of the new run.
    """
    if not os.path.isdir(root):
        raise FileNotFoundError(f"dataset directory not found: {root}")
    listed: list[str] | None = None
    manifest_path = os.path.join(root, MANIFEST)
    if os.path.isfile(manifest_path):
        with open(manifest_path, encoding="utf-8") as handle:
            listed = list(json.load(handle).get("cases", []))

    cases: list[Case] = []
    for name in sorted(os.listdir(root)):
        directory = os.path.join(root, name)
        meta_path = os.path.join(directory, "metadata.json")
        if not os.path.isfile(meta_path):
            continue
        with open(meta_path, encoding="utf-8") as handle:
            metadata = json.load(handle)
        case_id = metadata.get("id", name)
        if listed is not None and case_id not in listed:
            continue
        cases.append(Case(case_id=case_id, directory=directory, metadata=metadata))
    if not cases:
        raise FileNotFoundError(f"no cases with metadata.json under {root}")
    if listed is not None and len(cases) != len(listed):
        missing = sorted(set(listed) - {case.case_id for case in cases})
        raise FileNotFoundError(f"{root}: manifest lists cases that are missing: {missing[:5]}")
    return cases
