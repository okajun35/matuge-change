"""Brush strokes as values: painted vectors instead of a flattened raster.

Keeping strokes as vectors lets a session be reopened with its undo history and
turns the manual corrections into reusable training data.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from enum import Enum
from typing import Any

import cv2
import numpy as np


class BrushTool(Enum):
    """Brush kinds, with the trimap constraint value each one paints."""

    FOREGROUND = ("fg", 1)
    UNKNOWN = ("unknown", 2)
    BACKGROUND = ("bg", -1)

    def __init__(self, label: str, constraint: int) -> None:
        self.label = label
        self.constraint = constraint

    @classmethod
    def of(cls, label: str) -> BrushTool:
        for tool in cls:
            if tool.label == label:
                return tool
        raise ValueError(f"unknown brush tool: {label!r}")


@dataclass(frozen=True)
class Stroke:
    tool: BrushTool
    radius: int
    points: list[tuple[float, float]]

    def __post_init__(self) -> None:
        if self.radius <= 0:
            raise ValueError("stroke radius must be positive")
        if not self.points:
            raise ValueError("stroke must contain at least one point")

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> Stroke:
        try:
            tool, radius, points = raw["tool"], raw["radius"], raw["points"]
        except (KeyError, TypeError) as exc:
            raise ValueError(f"malformed stroke: {raw!r}") from exc
        return cls(
            tool=BrushTool.of(tool),
            radius=int(radius),
            points=[(float(p[0]), float(p[1])) for p in points],
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "tool": self.tool.label,
            "radius": self.radius,
            "points": [
                [int(x) if x.is_integer() else x, int(y) if y.is_integer() else y] for x, y in self.points
            ],
        }

    def draw(self, canvas: np.ndarray) -> None:
        pts = np.array(self.points, np.int32)
        cv2.polylines(canvas, [pts], False, self.tool.constraint, thickness=self.radius * 2)
        for x, y in pts:
            cv2.circle(canvas, (int(x), int(y)), self.radius, self.tool.constraint, -1)


@dataclass(frozen=True)
class StrokeSet:
    width: int
    height: int
    strokes: list[Stroke]

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise ValueError("canvas size must be positive")

    def __len__(self) -> int:
        return len(self.strokes)

    def __iter__(self) -> Iterator[Stroke]:
        return iter(self.strokes)

    @classmethod
    def from_payload(cls, width: int, height: int, payload: list[dict[str, Any]]) -> StrokeSet:
        return cls(int(width), int(height), [Stroke.from_dict(raw) for raw in payload])

    def to_payload(self) -> list[dict[str, Any]]:
        return [s.to_dict() for s in self.strokes]

    def rasterize(self) -> np.ndarray:
        """Constraint map for the trimap builder: +1 product, 2 unknown, -1 background."""
        canvas = np.zeros((self.height, self.width), np.int8)
        for stroke in self.strokes:
            stroke.draw(canvas)
        return canvas
