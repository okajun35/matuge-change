import json
import os
import shutil
import uuid

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from backend import video
from backend.app import DATA_DIR, app

client = TestClient(app)

N_LANDMARKS = 478

# EAR landmark indices (must match backend.video)
R_H = (33, 133)
R_V = [(159, 145), (158, 153)]
L_H = (362, 263)
L_V = [(386, 374), (385, 380)]


def make_landmarks(openness: float) -> np.ndarray:
    """Synthetic landmarks where both eyes have a given vertical/horizontal ratio."""
    lms = np.full((N_LANDMARKS, 2), 200.0)
    for (h0, h1), verts, cx in ((R_H, R_V, 100.0), (L_H, L_V, 300.0)):
        lms[h0] = [cx - 30, 200.0]
        lms[h1] = [cx + 30, 200.0]
        for top, bot in verts:
            half = 60.0 * openness / 2  # vertical opening = width * openness
            lms[top] = [cx, 200.0 - half]
            lms[bot] = [cx, 200.0 + half]
    return lms


class TestEyeOpenness:
    def test_open_eye_scores_higher_than_closed(self):
        assert video.eye_openness(make_landmarks(0.6)) > video.eye_openness(make_landmarks(0.05))

    def test_expected_ratio(self):
        assert video.eye_openness(make_landmarks(0.5)) == pytest.approx(0.5, abs=1e-6)


class TestSharpness:
    def test_sharp_image_scores_higher_than_blurred(self):
        rng = np.random.default_rng(3)
        sharp = rng.integers(0, 256, size=(64, 64, 3), dtype=np.uint8)
        blurred = cv2.GaussianBlur(sharp, (0, 0), 5)
        assert video.laplacian_sharpness(sharp) > video.laplacian_sharpness(blurred)


class TestSelectBestFrame:
    def test_prefers_sharp_and_open(self):
        rng = np.random.default_rng(4)
        sharp = rng.integers(0, 256, size=(64, 64, 3), dtype=np.uint8)
        blurred = cv2.GaussianBlur(sharp, (0, 0), 6)
        frames = [blurred, sharp, sharp, blurred]
        lms = [make_landmarks(0.6), make_landmarks(0.05), make_landmarks(0.6), make_landmarks(0.6)]
        assert video.select_best_frame(frames, lms) == 2

    def test_skips_frames_without_landmarks(self):
        rng = np.random.default_rng(5)
        img = rng.integers(0, 256, size=(64, 64, 3), dtype=np.uint8)
        frames = [img, img]
        assert video.select_best_frame(frames, [None, make_landmarks(0.5)]) == 1

    def test_raises_when_no_face_in_any_frame(self):
        img = np.zeros((32, 32, 3), np.uint8)
        with pytest.raises(ValueError):
            video.select_best_frame([img], [None])


class TestEyeRegionMask:
    def test_mask_range_and_coverage(self):
        lms = make_landmarks(0.5)
        mask = video.eye_region_mask((400, 400), lms)
        assert mask.shape == (400, 400)
        assert mask.min() >= 0.0 and mask.max() <= 1.0
        assert mask[200, 100] > 0.9  # inside right eye
        assert mask[200, 300] > 0.9  # inside left eye
        assert mask[390, 10] < 0.05  # far corner

    def test_expand_grows_mask(self):
        lms = make_landmarks(0.5)
        small = video.eye_region_mask((400, 400), lms, expand=0.2)
        big = video.eye_region_mask((400, 400), lms, expand=0.8)
        assert big.sum() > small.sum()


class TestBlendWithMask:
    def test_mask_extremes(self):
        frame = np.full((10, 10, 3), 200, np.uint8)
        edited = np.full((10, 10, 3), 50, np.uint8)
        mask = np.zeros((10, 10), np.float64)
        mask[:5] = 1.0
        out = video.blend_with_mask(frame, edited, mask)
        assert (out[:5] == 200).all()  # mask=1 keeps original frame
        assert (out[5:] == 50).all()  # mask=0 takes edited image


class TestWarpEditedToFrame:
    def test_identity_landmarks_keep_image(self):
        rng = np.random.default_rng(6)
        edited = rng.integers(0, 256, size=(400, 400, 3), dtype=np.uint8)
        lms = make_landmarks(0.5)
        warped = video.warp_edited_to_frame(edited, lms, lms, (400, 400))
        assert np.abs(warped.astype(int) - edited.astype(int)).mean() < 2.0

    def test_translation_is_applied(self):
        edited = np.zeros((400, 400, 3), np.uint8)
        edited[195:205, 95:105] = 255  # patch at right-eye centre
        lms_e = make_landmarks(0.5)
        lms_f = lms_e + [50.0, 0.0]
        warped = video.warp_edited_to_frame(edited, lms_e, lms_f, (400, 400))
        assert warped[200, 150 + 2].max() == 255
        assert warped[200, 100].max() == 0


class TestComposeFrames:
    def test_eye_region_from_frame_rest_from_edited(self):
        lms = make_landmarks(0.5)
        frame = np.full((400, 400, 3), 200, np.uint8)
        edited = np.full((400, 400, 3), 50, np.uint8)
        out = video.compose_frames([frame], [lms], edited, lms)[0]
        assert (out[200, 100] == 200).all()  # eye centre keeps original
        assert (out[390, 10] == 50).all()  # far corner becomes edited

    def test_frame_without_landmarks_reuses_previous(self):
        lms = make_landmarks(0.5)
        frame = np.full((400, 400, 3), 200, np.uint8)
        edited = np.full((400, 400, 3), 50, np.uint8)
        outs = video.compose_frames([frame, frame], [lms, None], edited, lms)
        assert (outs[0] == outs[1]).all()


class TestVideoRoundtrip:
    def test_write_and_read(self, tmp_path):
        rng = np.random.default_rng(11)
        frames = [rng.integers(0, 256, size=(48, 64, 3), dtype=np.uint8) for _ in range(8)]
        path = str(tmp_path / "out.mp4")
        video.write_video(frames, 20.0, path)
        assert os.path.exists(path)
        read, fps = video.read_video_frames(path)
        assert len(read) == 8
        assert fps == pytest.approx(20.0, rel=0.1)
        assert read[0].shape == (48, 64, 3)

    def test_read_missing_file_raises(self):
        with pytest.raises(ValueError):
            video.read_video_frames("/nonexistent/file.mp4")


def encode_png(img: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".png", img)
    assert ok
    return buf.tobytes()


def make_video_bytes(frames: list[np.ndarray], fps: float, tmp_path) -> bytes:
    path = str(tmp_path / "in.mp4")
    video.write_video(frames, fps, path)
    with open(path, "rb") as f:
        return f.read()


@pytest.fixture
def fake_video_session():
    """A video session dir bypassing face detection."""
    session_id = "test" + uuid.uuid4().hex[:8]
    sdir = os.path.join(DATA_DIR, session_id)
    fdir = os.path.join(sdir, "frames")
    os.makedirs(fdir)
    lms = make_landmarks(0.5)
    frames = 4
    for i in range(frames):
        cv2.imwrite(os.path.join(fdir, f"{i:06d}.png"), np.full((400, 400, 3), 200, np.uint8))
    all_lms = np.stack([lms] * frames)
    np.save(os.path.join(sdir, "video_landmarks.npy"), all_lms)
    with open(os.path.join(sdir, "meta.json"), "w") as f:
        json.dump(
            {
                "type": "video",
                "fps": 20.0,
                "frame_count": frames,
                "best_frame_index": 0,
                "width": 400,
                "height": 400,
            },
            f,
        )
    yield session_id
    shutil.rmtree(sdir, ignore_errors=True)


class TestVideoSessionApi:
    def test_undecodable_video_returns_400(self):
        res = client.post("/api/video/session", files={"video": ("a.mp4", b"not a video")})
        assert res.status_code == 400

    def test_no_face_returns_422(self, tmp_path):
        rng = np.random.default_rng(12)
        frames = [rng.integers(0, 256, size=(48, 64, 3), dtype=np.uint8) for _ in range(3)]
        data = make_video_bytes(frames, 10.0, tmp_path)
        res = client.post("/api/video/session", files={"video": ("a.mp4", data)})
        assert res.status_code == 422
        assert "no face" in res.json()["detail"]


class TestVideoComposeApi:
    def test_unknown_session_returns_404(self):
        res = client.post(
            "/api/video/compose",
            data={"session_id": "nonexistent"},
            files={"edited_image": ("e.png", encode_png(np.zeros((32, 32, 3), np.uint8)))},
        )
        assert res.status_code == 404

    def test_non_video_session_returns_409(self, fake_video_session):
        sdir = os.path.join(DATA_DIR, fake_video_session)
        with open(os.path.join(sdir, "meta.json")) as f:
            meta = json.load(f)
        meta["type"] = "image"
        with open(os.path.join(sdir, "meta.json"), "w") as f:
            json.dump(meta, f)
        res = client.post(
            "/api/video/compose",
            data={"session_id": fake_video_session},
            files={"edited_image": ("e.png", encode_png(np.zeros((32, 32, 3), np.uint8)))},
        )
        assert res.status_code == 409

    def test_no_face_in_edited_returns_422(self, fake_video_session):
        noise = np.random.default_rng(13).integers(0, 256, size=(64, 64, 3), dtype=np.uint8)
        res = client.post(
            "/api/video/compose",
            data={"session_id": fake_video_session},
            files={"edited_image": ("e.png", encode_png(noise))},
        )
        assert res.status_code == 422
        assert "no face" in res.json()["detail"]


class TestVideoServing:
    def test_output_not_ready_returns_404(self, fake_video_session):
        res = client.get(f"/api/video/{fake_video_session}/output")
        assert res.status_code == 404

    def test_output_served_when_present(self, fake_video_session, tmp_path):
        sdir = os.path.join(DATA_DIR, fake_video_session)
        frames = [np.zeros((48, 64, 3), np.uint8) for _ in range(2)]
        video.write_video(frames, 10.0, os.path.join(sdir, "output.mp4"))
        res = client.get(f"/api/video/{fake_video_session}/output")
        assert res.status_code == 200
        assert res.headers["content-type"] == "video/mp4"

    def test_bad_name_returns_400(self, fake_video_session):
        res = client.get(f"/api/video/{fake_video_session}/bad-name!")
        assert res.status_code == 400
