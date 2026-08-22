"""Video mode endpoints: best-frame selection and per-frame eye-region compositing."""

from __future__ import annotations

import os
import tempfile
from collections.abc import Callable

import cv2
import numpy as np
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse

from backend import video
from backend.api.container import container
from backend.api.errors import read_upload, to_http
from backend.lash_extraction import detect_landmarks
from backend.video_jobs import VideoJobRunner

router = APIRouter(prefix="/api/video")

N_LANDMARKS = 478
video_jobs = VideoJobRunner()
VideoReport = Callable[[str, int, int | None, int | None, str], None]


def _frame_path(session_id: str, index: int) -> str:
    return container().store.path(session_id, "frames", f"{index:06d}.png")


def _no_progress(_phase: str, _progress: int, _done: int | None, _total: int | None, _message: str) -> None:
    """Adapter for the synchronous compatibility endpoints."""


def _analyze_video(
    session_id: str,
    path: str,
    report: VideoReport,
    *,
    select_best: bool,
) -> dict:
    """Decode, track, and persist a video session, reporting real frame progress."""
    report("decode", 5, None, None, "フレームを読み込んでいます")
    frames, fps = video.read_video_frames(path)
    total = len(frames)
    landmarks = []
    for index, frame in enumerate(frames, start=1):
        landmarks.append(detect_landmarks(frame))
        report("detect_landmarks", 10 + int(50 * index / total), index, total, "顔と目元を検出しています")
    valid = [index for index, lms in enumerate(landmarks) if lms is not None]
    if not valid:
        raise ValueError("no face detected in the video")

    best = video.select_best_frame(frames, landmarks) if select_best else None
    report("save_session", 65, total, total, "解析結果を保存しています")
    store = container().store
    os.makedirs(store.path(session_id, "frames"), exist_ok=True)
    for index, frame in enumerate(frames):
        cv2.imwrite(_frame_path(session_id, index), frame)
    stacked = np.stack([lms if lms is not None else np.full((N_LANDMARKS, 2), np.nan) for lms in landmarks])
    store.save_array(session_id, "video_landmarks", stacked)
    if best is not None:
        store.save_image(session_id, "best_frame", frames[best])
    h, w = frames[0].shape[:2]
    store.save_meta(
        session_id,
        {
            "type": "video",
            "fps": fps,
            "frame_count": total,
            "best_frame_index": best,
            "width": w,
            "height": h,
        },
    )
    return {
        "session_id": session_id,
        "frame_count": total,
        "fps": fps,
        "best_frame_index": best,
        "width": w,
        "height": h,
        "has_best_frame": best is not None,
    }


def _compose_video(session_id: str, edited: np.ndarray, expand: float, report: VideoReport) -> dict:
    """Compose a persisted session and report the frame count while processing."""
    report("validate_edited", 68, None, None, "加工画像を確認しています")
    lms_edited = detect_landmarks(edited)
    if lms_edited is None:
        raise ValueError("no face detected in the edited image")

    store = container().store
    meta = store.load_meta(session_id)
    frames = [cv2.imread(_frame_path(session_id, index)) for index in range(meta["frame_count"])]
    stacked = store.load_array(session_id, "video_landmarks")
    landmarks: list[np.ndarray | None] = [None if np.isnan(lms).any() else lms for lms in stacked]

    def on_frame(done: int, total: int) -> None:
        report("compose", 70 + int(22 * done / total), done, total, "各フレームを合成しています")

    outs = video.compose_frames(frames, landmarks, edited, lms_edited, expand, meta["fps"], on_frame)
    report("encode", 93, meta["frame_count"], meta["frame_count"], "動画を書き出しています")
    video.write_video(outs, meta["fps"], store.path(session_id, "output.mp4"))
    return {"session_id": session_id, "frame_count": len(outs), "composed": True}


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


@router.post("/jobs", status_code=202)
async def start_video_job(
    video_file: UploadFile = File(..., alias="video"),
    edited_image: UploadFile | None = File(None),
    expand: float = Form(0.45),
):
    """Queue analysis, and compose immediately when an edited image is already supplied."""
    store = container().store
    session_id = store.create()
    video_path = store.path(session_id, "input.mp4")
    with open(video_path, "wb") as destination:
        destination.write(video_file.file.read())

    edited = read_upload(edited_image) if edited_image is not None else None

    def work(report: VideoReport) -> dict:
        analyzed = _analyze_video(session_id, video_path, report, select_best=edited is None)
        if edited is None:
            return analyzed
        composed = _compose_video(session_id, edited, expand, report)
        return analyzed | composed

    job_id = video_jobs.submit(session_id, work)
    return {"job_id": job_id, "session_id": session_id}


@router.post("/{session_id}/compose/jobs", status_code=202)
async def start_compose_job(
    session_id: str,
    edited_image: UploadFile = File(...),
    expand: float = Form(0.45),
):
    """Queue a recompose of an already analyzed video session."""
    store = container().store
    try:
        store.require(session_id)
        meta = store.load_meta(session_id)
    except Exception as exc:
        raise to_http(exc) from exc
    if meta.get("type") != "video":
        raise HTTPException(409, "not a video session")
    edited = read_upload(edited_image)
    job_id = video_jobs.submit(session_id, lambda report: _compose_video(session_id, edited, expand, report))
    return {"job_id": job_id, "session_id": session_id}


@router.get("/jobs/{job_id}")
async def get_video_job(job_id: str):
    job = video_jobs.get(job_id)
    if job is None:
        raise HTTPException(404, "video job not found")
    return job.to_dict()


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
    outs = video.compose_frames(frames, landmarks, edited, lms_edited, expand, meta["fps"])
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
