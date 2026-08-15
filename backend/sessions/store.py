"""Filesystem layout of a working session (ROI, probability, layers, meta)."""

from __future__ import annotations

import json
import os
import uuid
from typing import Any

import cv2
import numpy as np

from backend.sessions.errors import SessionNotFound


class SessionStore:
    def __init__(self, root: str) -> None:
        self.root = root
        os.makedirs(root, exist_ok=True)

    def create(self) -> str:
        session_id = uuid.uuid4().hex[:12]
        os.makedirs(os.path.join(self.root, session_id))
        return session_id

    def path(self, session_id: str, *parts: str) -> str:
        return os.path.join(self.root, session_id, *parts)

    def exists(self, session_id: str) -> bool:
        return os.path.isdir(os.path.join(self.root, session_id))

    def require(self, session_id: str) -> str:
        if not self.exists(session_id):
            raise SessionNotFound(session_id)
        return os.path.join(self.root, session_id)

    def has_layer(self, session_id: str, name: str) -> bool:
        return os.path.exists(self.path(session_id, f"{name}.png"))

    def has_array(self, session_id: str, name: str) -> bool:
        return os.path.exists(self.path(session_id, f"{name}.npy"))

    def layer_path(self, session_id: str, name: str) -> str:
        return self.path(session_id, f"{name}.png")

    def save_image(self, session_id: str, name: str, img: np.ndarray) -> None:
        cv2.imwrite(self.layer_path(session_id, name), img)

    def load_image(self, session_id: str, name: str, flags: int = cv2.IMREAD_COLOR) -> np.ndarray:
        return cv2.imread(self.layer_path(session_id, name), flags)

    def save_gray(self, session_id: str, name: str, x: np.ndarray) -> None:
        self.save_image(session_id, name, (np.clip(x, 0, 1) * 255).astype(np.uint8))

    def save_array(self, session_id: str, name: str, array: np.ndarray) -> None:
        np.save(self.path(session_id, f"{name}.npy"), array)

    def load_array(self, session_id: str, name: str) -> np.ndarray:
        return np.load(self.path(session_id, f"{name}.npy"))

    def append_run(self, session_id: str, run: dict[str, Any]) -> None:
        runs = self.load_runs(session_id)
        runs.append(run)
        with open(self.path(session_id, "runs.json"), "w") as f:
            json.dump(runs, f)

    def load_runs(self, session_id: str) -> list[dict[str, Any]]:
        path = self.path(session_id, "runs.json")
        if not os.path.exists(path):
            return []
        with open(path) as f:
            return json.load(f)

    def save_meta(self, session_id: str, meta: dict[str, Any]) -> None:
        with open(self.path(session_id, "meta.json"), "w") as f:
            json.dump(meta, f)

    def load_meta(self, session_id: str) -> dict[str, Any]:
        with open(self.path(session_id, "meta.json")) as f:
            return json.load(f)
