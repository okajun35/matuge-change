import json
import os
import shutil
import uuid

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from backend.app import DATA_DIR, app

N_LANDMARKS = 478


@pytest.fixture
def synthetic_landmarks() -> np.ndarray:
    """478 face-mesh-like landmark points spread over a 400x400 image."""
    rng = np.random.default_rng(42)
    return rng.uniform(50, 350, size=(N_LANDMARKS, 2)).astype(np.float64)


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


@pytest.fixture
def session_id():
    """A session dir with a synthetic ROI + probability, bypassing face detection."""
    sid = "test" + uuid.uuid4().hex[:8]
    sdir = os.path.join(DATA_DIR, sid)
    os.makedirs(sdir)
    rng = np.random.default_rng(7)
    roi_a = rng.integers(0, 256, size=(48, 48, 3), dtype=np.uint8)
    prob = np.zeros((48, 48), np.float64)
    prob[20:28, 20:28] = 0.9
    cv2.imwrite(os.path.join(sdir, "roi_a.png"), roi_a)
    np.save(os.path.join(sdir, "probability.npy"), prob)
    np.save(os.path.join(sdir, "landmarks.npy"), rng.uniform(0, 48, size=(478, 2)))
    with open(os.path.join(sdir, "meta.json"), "w") as f:
        json.dump({"roi": [0, 0, 48, 48], "scale": 1.0, "has_bare": False, "width": 48, "height": 48}, f)
    yield sid
    shutil.rmtree(sdir, ignore_errors=True)


class InMemoryArchive:
    """Supabase Storage の代わりに使うオブジェクトストア。"""

    def __init__(self) -> None:
        self.objects: dict[str, bytes] = {}

    def upload(self, path: str, data: bytes) -> None:
        self.objects[path] = data

    def download(self, path: str) -> bytes:
        return self.objects[path]

    def list(self, prefix: str) -> list[str]:
        return sorted(p for p in self.objects if p.startswith(f"{prefix}/"))


@pytest.fixture
def fake_archive():
    """Supabase 未設定の環境でも退避／復元エンドポイントを検証できるようにする。"""
    from backend.api.container import container

    archive = InMemoryArchive()
    service = container().archive
    original = service.archive
    service.archive = archive
    yield archive
    service.archive = original


@pytest.fixture
def matted_session(client, session_id) -> str:
    assert client.post("/api/matte", data={"session_id": session_id}).status_code == 200
    return session_id
