import base64
import json
import os
import shutil
import uuid

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from backend.app import DATA_DIR, app

client = TestClient(app)


def encode_png(img: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".png", img)
    assert ok
    return buf.tobytes()


def png_data_url(img: np.ndarray) -> str:
    return "data:image/png;base64," + base64.b64encode(encode_png(img)).decode()


@pytest.fixture
def fake_session():
    """A session dir with a synthetic ROI + probability, bypassing face detection."""
    session_id = "test" + uuid.uuid4().hex[:8]
    sdir = os.path.join(DATA_DIR, session_id)
    os.makedirs(sdir)
    rng = np.random.default_rng(7)
    roi_a = rng.integers(0, 256, size=(48, 48, 3), dtype=np.uint8)
    prob = np.zeros((48, 48), np.float64)
    prob[20:28, 20:28] = 0.9
    cv2.imwrite(os.path.join(sdir, "roi_a.png"), roi_a)
    np.save(os.path.join(sdir, "probability.npy"), prob)
    lms = rng.uniform(0, 48, size=(478, 2))
    np.save(os.path.join(sdir, "landmarks.npy"), lms)
    with open(os.path.join(sdir, "meta.json"), "w") as f:
        json.dump({"roi": [0, 0, 48, 48], "scale": 1.0, "has_bare": False, "width": 48, "height": 48}, f)
    yield session_id
    shutil.rmtree(sdir, ignore_errors=True)


class TestSession:
    def test_no_face_returns_422(self):
        noise = np.random.default_rng(8).integers(0, 256, size=(64, 64, 3), dtype=np.uint8)
        res = client.post("/api/session", files={"image_with": ("a.png", encode_png(noise))})
        assert res.status_code == 422
        assert "no face" in res.json()["detail"]

    def test_undecodable_image_returns_400(self):
        res = client.post("/api/session", files={"image_with": ("a.png", b"not an image")})
        assert res.status_code == 400


class TestMatte:
    def test_unknown_session_returns_404(self):
        res = client.post("/api/matte", data={"session_id": "nonexistent"})
        assert res.status_code == 404

    def test_matte_without_constraints(self, fake_session):
        res = client.post("/api/matte", data={"session_id": fake_session})
        assert res.status_code == 200
        body = res.json()
        assert "trimap" in body["layers"]
        assert "product_rgba" in body["layers"]
        assert "composite_on_bare" not in body["layers"]  # no bare image
        assert isinstance(body["reconstruction_error"], float)
        sdir = os.path.join(DATA_DIR, fake_session)
        assert os.path.exists(os.path.join(sdir, "product_rgba.png"))

    def test_constraint_colors_map_to_trimap(self, fake_session):
        c = np.zeros((48, 48, 4), np.uint8)
        c[0:8, 0:8] = (32, 32, 255, 140)  # red -> FG
        c[8:16, 0:8] = (32, 192, 64, 140)  # green -> unknown
        c[16:24, 0:8] = (255, 64, 32, 140)  # blue -> BG
        res = client.post(
            "/api/matte",
            data={"session_id": fake_session, "constraints_png": png_data_url(c)},
        )
        assert res.status_code == 200
        trimap = cv2.imread(os.path.join(DATA_DIR, fake_session, "trimap.png"), cv2.IMREAD_GRAYSCALE)
        assert (trimap[0:8, 0:8] == 255).all()
        assert (trimap[8:16, 0:8] == 128).all()
        assert (trimap[16:24, 0:8] == 0).all()

    def test_wrong_size_constraints_returns_400(self, fake_session):
        c = np.zeros((10, 10, 4), np.uint8)
        res = client.post(
            "/api/matte",
            data={"session_id": fake_session, "constraints_png": png_data_url(c)},
        )
        assert res.status_code == 400


class TestRecompose:
    def test_unknown_session_returns_404(self):
        noise = np.zeros((32, 32, 3), np.uint8)
        res = client.post(
            "/api/recompose",
            data={"session_id": "nonexistent"},
            files={"edited_image": ("e.png", encode_png(noise))},
        )
        assert res.status_code == 404

    def test_before_matting_returns_409(self, fake_session):
        noise = np.zeros((32, 32, 3), np.uint8)
        res = client.post(
            "/api/recompose",
            data={"session_id": fake_session},
            files={"edited_image": ("e.png", encode_png(noise))},
        )
        assert res.status_code == 409
        assert "matting" in res.json()["detail"]

    def test_no_face_in_edited_returns_422(self, fake_session):
        client.post("/api/matte", data={"session_id": fake_session})
        noise = np.random.default_rng(9).integers(0, 256, size=(64, 64, 3), dtype=np.uint8)
        res = client.post(
            "/api/recompose",
            data={"session_id": fake_session},
            files={"edited_image": ("e.png", encode_png(noise))},
        )
        assert res.status_code == 422
        assert "no face" in res.json()["detail"]


class TestGetImage:
    def test_bad_layer_name_returns_400(self, fake_session):
        res = client.get(f"/api/image/{fake_session}/../secret")
        assert res.status_code in (400, 404)
        res = client.get(f"/api/image/{fake_session}/bad-name!")
        assert res.status_code == 400

    def test_unknown_layer_returns_404(self, fake_session):
        res = client.get(f"/api/image/{fake_session}/alpha")
        assert res.status_code == 404

    def test_existing_layer_served(self, fake_session):
        res = client.get(f"/api/image/{fake_session}/roi_a")
        assert res.status_code == 200
        assert res.headers["content-type"] == "image/png"


class TestPages:
    """カタログ / 静止画モード / 動画モードはページとして分かれている。"""

    def test_root_serves_catalog(self):
        res = client.get("/")
        assert res.status_code == 200
        assert "/api/assets" in res.text
        assert "/extract.html" in res.text
        assert "/video.html" in res.text

    def test_catalog_page_has_search_and_pagination(self):
        res = client.get("/")
        for marker in ("assetQuery", "btnPrevPage", "btnNextPage", "offset"):
            assert marker in res.text

    def test_catalog_page_offers_png_and_mask_downloads(self):
        res = client.get("/")
        assert "/image" in res.text
        assert "/mask" in res.text
        assert "download" in res.text

    def test_extract_page_has_static_mode_controls(self):
        res = client.get("/extract.html")
        assert res.status_code == 200
        for marker in ("fileWith", "btnMatte", "btnRecompose", "btnUndo", "/api/matte/jobs"):
            assert marker in res.text
        assert "/api/video/" not in res.text

    def test_video_page_has_video_mode_controls(self):
        res = client.get("/video.html")
        assert res.status_code == 200
        for marker in ("fileVideo", "btnCompose", "/api/video/session", "/api/video/compose"):
            assert marker in res.text
        assert "btnMatte" not in res.text

    def test_shared_stylesheet_served(self):
        res = client.get("/common.css")
        assert res.status_code == 200
        assert "text/css" in res.headers["content-type"]
