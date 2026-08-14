"""FastAPI app for the lash extraction PoC."""

from __future__ import annotations

import base64
import json
import os
import tempfile
import uuid

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from backend import pipeline, video

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data")
FRONTEND_DIR = os.path.join(ROOT, "frontend")
os.makedirs(DATA_DIR, exist_ok=True)

app = FastAPI(title="matuge-change PoC")


def session_dir(session_id: str) -> str:
    path = os.path.join(DATA_DIR, session_id)
    if not os.path.isdir(path):
        raise HTTPException(404, "session not found")
    return path


def imwrite(path: str, img: np.ndarray) -> None:
    cv2.imwrite(path, img)


def gray_png(x: np.ndarray) -> np.ndarray:
    return (np.clip(x, 0, 1) * 255).astype(np.uint8)


def read_upload(file: UploadFile) -> np.ndarray:
    data = np.frombuffer(file.file.read(), np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(400, f"could not decode image: {file.filename}")
    return img


@app.post("/api/session")
async def create_session(
    image_with: UploadFile = File(...),
    image_without: UploadFile | None = File(None),
):
    img_a = read_upload(image_with)
    img_b = read_upload(image_without) if image_without is not None else None

    lms_a = pipeline.detect_landmarks(img_a)
    if lms_a is None:
        raise HTTPException(422, "no face detected in the worn image")

    roi = pipeline.compute_eye_roi(lms_a, img_a.shape)
    roi_a = pipeline.crop_roi(img_a, roi)

    if img_b is not None:
        lms_b = pipeline.detect_landmarks(img_b)
        if lms_b is None:
            raise HTTPException(422, "no face detected in the bare image")
        warped_b = pipeline.align_b_to_a(img_a, lms_a, img_b, lms_b)
        roi_b = pipeline.crop_roi(warped_b, roi)
        roi_b = pipeline.ecc_refine(roi_a, roi_b)
        evidence = pipeline.difference_map(roi_a, roi_b)
    else:
        roi_b = None
        evidence = pipeline.darkness_map(roi_a)

    prior = pipeline.eye_prior(roi_a.shape, lms_a, roi)
    prob = pipeline.initial_probability(evidence, prior)

    session_id = uuid.uuid4().hex[:12]
    sdir = os.path.join(DATA_DIR, session_id)
    os.makedirs(sdir)
    imwrite(os.path.join(sdir, "roi_a.png"), roi_a)
    if roi_b is not None:
        imwrite(os.path.join(sdir, "roi_b.png"), roi_b)
    imwrite(os.path.join(sdir, "difference.png"), gray_png(evidence))
    imwrite(os.path.join(sdir, "probability.png"), gray_png(prob))
    np.save(os.path.join(sdir, "probability.npy"), prob)
    np.save(os.path.join(sdir, "landmarks.npy"), lms_a)
    with open(os.path.join(sdir, "meta.json"), "w") as f:
        json.dump(
            {
                "roi": [roi.x0, roi.y0, roi.x1, roi.y1],
                "scale": roi.scale,
                "has_bare": roi_b is not None,
                "width": roi_a.shape[1],
                "height": roi_a.shape[0],
            },
            f,
        )

    return {
        "session_id": session_id,
        "width": roi_a.shape[1],
        "height": roi_a.shape[0],
        "has_bare": roi_b is not None,
        "layers": ["roi_a"] + (["roi_b"] if roi_b is not None else []) + ["difference", "probability"],
    }


@app.post("/api/matte")
async def run_matte(
    session_id: str = Form(...),
    constraints_png: str = Form(""),
    fg_thresh: float = Form(0.70),
    bg_thresh: float = Form(0.18),
    unknown_band_px: int = Form(6),
):
    sdir = session_dir(session_id)
    roi_a = cv2.imread(os.path.join(sdir, "roi_a.png"))
    prob = np.load(os.path.join(sdir, "probability.npy"))

    constraints = None
    if constraints_png:
        raw = base64.b64decode(constraints_png.split(",")[-1])
        cimg = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_UNCHANGED)
        if cimg is None or cimg.shape[:2] != prob.shape:
            raise HTTPException(400, "invalid constraints image")
        # UI encodes on transparent canvas: red = product (+1),
        # green = unknown (2), blue = background (-1)
        alpha_ch = cimg[..., 3] if cimg.shape[2] == 4 else np.full(prob.shape, 255, np.uint8)
        constraints = np.zeros(prob.shape, np.int8)
        painted = alpha_ch > 64
        b = cimg[..., 0].astype(int)
        g = cimg[..., 1].astype(int)
        r = cimg[..., 2].astype(int)
        constraints[painted & (r >= g) & (r >= b)] = 1
        constraints[painted & (g > r) & (g >= b)] = 2
        constraints[painted & (b > r) & (b > g)] = -1

    trimap = pipeline.build_trimap(prob, constraints, fg_thresh, bg_thresh, unknown_band_px)
    alpha, fg = pipeline.run_matting(roi_a, trimap)

    rgba = np.dstack([(fg * 255).astype(np.uint8), (alpha * 255).astype(np.uint8)])
    imwrite(os.path.join(sdir, "trimap.png"), trimap)
    imwrite(os.path.join(sdir, "alpha.png"), gray_png(alpha))
    imwrite(os.path.join(sdir, "product_rgba.png"), rgba)

    recon_err = pipeline.reconstruction_error(alpha, fg, roi_a)
    layers = ["trimap", "alpha", "product_rgba"]

    roi_b_path = os.path.join(sdir, "roi_b.png")
    if os.path.exists(roi_b_path):
        roi_b = cv2.imread(roi_b_path)
        imwrite(os.path.join(sdir, "composite_on_bare.png"), pipeline.composite(alpha, fg, roi_b))
        layers.append("composite_on_bare")

    return {"layers": layers, "reconstruction_error": recon_err}


@app.post("/api/recompose")
async def recompose(
    session_id: str = Form(...),
    edited_image: UploadFile = File(...),
):
    sdir = session_dir(session_id)
    rgba_path = os.path.join(sdir, "product_rgba.png")
    if not os.path.exists(rgba_path):
        raise HTTPException(409, "run matting first")
    rgba = cv2.imread(rgba_path, cv2.IMREAD_UNCHANGED)
    lms_worn = np.load(os.path.join(sdir, "landmarks.npy"))
    with open(os.path.join(sdir, "meta.json")) as f:
        meta = json.load(f)
    x0, y0, x1, y1 = meta["roi"]
    roi = pipeline.EyeRoi(x0, y0, x1, y1, meta["scale"])
    edited = read_upload(edited_image)
    out = pipeline.recompose_onto(rgba, roi, lms_worn, edited)
    if out is None:
        raise HTTPException(422, "no face detected in the edited image")
    imwrite(os.path.join(sdir, "composite_on_edited.png"), out)
    return {"layers": ["composite_on_edited"]}


@app.post("/api/video/session")
async def create_video_session(video_file: UploadFile = File(..., alias="video")):
    data = video_file.file.read()
    with tempfile.NamedTemporaryFile(suffix=".mp4") as tmp:
        tmp.write(data)
        tmp.flush()
        try:
            frames, fps = video.read_video_frames(tmp.name)
        except ValueError as e:
            raise HTTPException(400, f"could not decode video: {video_file.filename}") from e

    landmarks = [pipeline.detect_landmarks(f) for f in frames]
    try:
        best = video.select_best_frame(frames, landmarks)
    except ValueError as e:
        raise HTTPException(422, "no face detected in the video") from e

    session_id = uuid.uuid4().hex[:12]
    sdir = os.path.join(DATA_DIR, session_id)
    fdir = os.path.join(sdir, "frames")
    os.makedirs(fdir)
    for i, frame in enumerate(frames):
        imwrite(os.path.join(fdir, f"{i:06d}.png"), frame)
    stacked = np.stack([lms if lms is not None else np.full((478, 2), np.nan) for lms in landmarks])
    np.save(os.path.join(sdir, "video_landmarks.npy"), stacked)
    imwrite(os.path.join(sdir, "best_frame.png"), frames[best])
    h, w = frames[0].shape[:2]
    with open(os.path.join(sdir, "meta.json"), "w") as f:
        json.dump(
            {
                "type": "video",
                "fps": fps,
                "frame_count": len(frames),
                "best_frame_index": best,
                "width": w,
                "height": h,
            },
            f,
        )

    return {
        "session_id": session_id,
        "frame_count": len(frames),
        "fps": fps,
        "best_frame_index": best,
        "width": w,
        "height": h,
        "layers": ["best_frame"],
    }


@app.post("/api/video/compose")
async def compose_video(
    session_id: str = Form(...),
    edited_image: UploadFile = File(...),
    expand: float = Form(0.45),
):
    sdir = session_dir(session_id)
    with open(os.path.join(sdir, "meta.json")) as f:
        meta = json.load(f)
    if meta.get("type") != "video":
        raise HTTPException(409, "not a video session")

    edited = read_upload(edited_image)
    lms_edited = pipeline.detect_landmarks(edited)
    if lms_edited is None:
        raise HTTPException(422, "no face detected in the edited image")

    fdir = os.path.join(sdir, "frames")
    stacked = np.load(os.path.join(sdir, "video_landmarks.npy"))
    frames = [cv2.imread(os.path.join(fdir, f"{i:06d}.png")) for i in range(meta["frame_count"])]
    landmarks: list[np.ndarray | None] = [None if np.isnan(lms).any() else lms for lms in stacked]
    outs = video.compose_frames(frames, landmarks, edited, lms_edited, expand)
    video.write_video(outs, meta["fps"], os.path.join(sdir, "output.mp4"))
    return {"video": "output", "frame_count": len(outs)}


@app.get("/api/video/{session_id}/{name}")
async def get_video(session_id: str, name: str):
    if not name.replace("_", "").isalnum():
        raise HTTPException(400, "bad video name")
    path = os.path.join(session_dir(session_id), f"{name}.mp4")
    if not os.path.exists(path):
        raise HTTPException(404, "video not found")
    return FileResponse(path, media_type="video/mp4")


@app.get("/api/image/{session_id}/{name}")
async def get_image(session_id: str, name: str):
    if not name.replace("_", "").isalnum():
        raise HTTPException(400, "bad layer name")
    path = os.path.join(session_dir(session_id), f"{name}.png")
    if not os.path.exists(path):
        raise HTTPException(404, "layer not found")
    return FileResponse(path, media_type="image/png")


app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
