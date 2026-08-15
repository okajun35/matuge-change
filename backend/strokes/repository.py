"""Stroke persistence ports and adapters."""

from __future__ import annotations

import json
import os
from typing import Any, Protocol


class StrokeRepository(Protocol):
    def save(self, session_id: str, width: int, height: int, payload: list[dict[str, Any]]) -> None: ...

    def load(self, session_id: str) -> dict[str, Any] | None: ...


class FileStrokeRepository:
    """Stores strokes next to the session artefacts (offline fallback)."""

    def __init__(self, data_dir: str) -> None:
        self._data_dir = data_dir

    def _path(self, session_id: str) -> str:
        return os.path.join(self._data_dir, session_id, "strokes.json")

    def save(self, session_id: str, width: int, height: int, payload: list[dict[str, Any]]) -> None:
        path = self._path(session_id)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump({"width": width, "height": height, "strokes": payload}, f)

    def load(self, session_id: str) -> dict[str, Any] | None:
        path = self._path(session_id)
        if not os.path.exists(path):
            return None
        with open(path) as f:
            return json.load(f)
