"""Video mode endpoints: best-frame selection and per-frame eye-region compositing."""

from __future__ import annotations

import os
import tempfile

import cv2
import numpy as np
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse

from backend import video
from backend.api.container import container
from backend.api.errors import read_upload, to_http
from backend.lash_extraction import detect_landmarks

router = APIRouter(prefix="/api/video")

N_LANDMARKS = 478


def _frame_path(session_id: str, index: int) -> str:
    return container().store.path(session_id, "frames", f"{index:06d}.png")


@router.post("/session")
async def create_video_session(video_file: UploadFile = File(..., alias="video")):
    data = video_file.file.read()
    with tempfile.NamedTemporaryFile(suffix=".mp4") as tmp:
        tmp.write(data)
        tmp.flush()
        try:
            frames, fps = video.read_video_frames(tmp.name)
        except ValueError as e:
            raise HTTPException(400, f"could not decode video: {video_file.filename}") from e

    landmarks = [detect_landmarks(f) for f in frames]
    try:
        best = video.select_best_frame(frames, landmarks)
    except ValueError as e:
        raise HTTPException(422, "no face detected in the video") from e

    store = container().store
    session_id = store.create()
    os.makedirs(store.path(session_id, "frames"))
    for i, frame in enumerate(frames):
        cv2.imwrite(_frame_path(session_id, i), frame)
    stacked = np.stack([lms if lms is not None else np.full((N_LANDMARKS, 2), np.nan) for lms in landmarks])
    store.save_array(session_id, "video_landmarks", stacked)
    store.save_image(session_id, "best_frame", frames[best])
    h, w = frames[0].shape[:2]
    store.save_meta(
        session_id,
        {
            "type": "video",
            "fps": fps,
            "frame_count": len(frames),
            "best_frame_index": best,
            "width": w,
            "height": h,
        },
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


@router.post("/compose")
async def compose_video(
    session_id: str = Form(...),
    edited_image: UploadFile = File(...),
    expand: float = Form(0.45),
):
    store = container().store
    try:
        store.require(session_id)
        meta = store.load_meta(session_id)
    except Exception as exc:
        raise to_http(exc) from exc
    if meta.get("type") != "video":
        raise HTTPException(409, "not a video session")

    edited = read_upload(edited_image)
    lms_edited = detect_landmarks(edited)
    if lms_edited is None:
        raise HTTPException(422, "no face detected in the edited image")

    stacked = store.load_array(session_id, "video_landmarks")
    frames = [cv2.imread(_frame_path(session_id, i)) for i in range(meta["frame_count"])]
    landmarks: list[np.ndarray | None] = [None if np.isnan(lms).any() else lms for lms in stacked]
    outs = video.compose_frames(frames, landmarks, edited, lms_edited, expand)
    video.write_video(outs, meta["fps"], store.path(session_id, "output.mp4"))
    return {"video": "output", "frame_count": len(outs)}


@router.get("/{session_id}/{name}")
async def get_video(session_id: str, name: str):
    if not name.replace("_", "").isalnum():
        raise HTTPException(400, "bad video name")
    store = container().store
    try:
        store.require(session_id)
    except Exception as exc:
        raise to_http(exc) from exc
    path = store.path(session_id, f"{name}.mp4")
    if not os.path.exists(path):
        raise HTTPException(404, "video not found")
    return FileResponse(path, media_type="video/mp4")
