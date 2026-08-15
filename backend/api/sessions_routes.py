"""Session, matting and recomposition endpoints."""

from __future__ import annotations

import numpy as np
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse

from backend.api.container import container
from backend.api.errors import read_upload, to_http
from backend.strokes.constraints import decode_constraints_png

router = APIRouter(prefix="/api")


@router.post("/session")
async def create_session(
    image_with: UploadFile = File(...),
    image_without: UploadFile | None = File(None),
):
    img_a = read_upload(image_with)
    img_b = read_upload(image_without) if image_without is not None else None
    try:
        return container().sessions.create(img_a, img_b)
    except Exception as exc:
        raise to_http(exc) from exc


def _constraints(session_id: str, constraints_png: str, use_saved_strokes: bool) -> np.ndarray | None:
    app = container()
    if use_saved_strokes:
        saved = app.strokes.load(session_id)
        return saved.rasterize() if saved else None
    if not constraints_png:
        return None
    return decode_constraints_png(constraints_png, app.sessions.probability_shape(session_id))


LAYER_ORDER = (
    "roi_a",
    "roi_b",
    "difference",
    "probability",
    "trimap",
    "alpha",
    "product_rgba",
    "composite_on_bare",
    "composite_on_edited",
)


@router.get("/sessions/{session_id}")
async def get_session(session_id: str):
    store = container().store
    try:
        store.require(session_id)
        meta = store.load_meta(session_id)
    except Exception as exc:
        raise to_http(exc) from exc
    return {
        "session_id": session_id,
        "width": meta["width"],
        "height": meta["height"],
        "has_bare": meta["has_bare"],
        "layers": [name for name in LAYER_ORDER if store.has_layer(session_id, name)],
    }


@router.post("/matte")
async def run_matte(
    session_id: str = Form(...),
    constraints_png: str = Form(""),
    use_saved_strokes: bool = Form(False),
    fg_thresh: float = Form(0.70),
    bg_thresh: float = Form(0.18),
    unknown_band_px: int = Form(6),
):
    app = container()
    try:
        app.sessions.store.require(session_id)
        constraints = _constraints(session_id, constraints_png, use_saved_strokes)
        return app.sessions.run_matte(session_id, constraints, fg_thresh, bg_thresh, unknown_band_px)
    except Exception as exc:
        raise to_http(exc) from exc


@router.post("/matte/jobs", status_code=202)
async def enqueue_matte(
    session_id: str = Form(...),
    constraints_png: str = Form(""),
    use_saved_strokes: bool = Form(False),
    fg_thresh: float = Form(0.70),
    bg_thresh: float = Form(0.18),
    unknown_band_px: int = Form(6),
):
    app = container()
    params = {
        "fg_thresh": fg_thresh,
        "bg_thresh": bg_thresh,
        "unknown_band_px": unknown_band_px,
        "use_saved_strokes": use_saved_strokes,
    }

    def work(report):
        app.sessions.store.require(session_id)
        constraints = _constraints(session_id, constraints_png, use_saved_strokes)
        return app.sessions.run_matte(
            session_id, constraints, fg_thresh, bg_thresh, unknown_band_px, report=report
        )

    job_id = app.jobs.submit(session_id, params, work)
    return {"job_id": job_id, "status": "queued"}


@router.get("/matte/jobs/{job_id}")
async def get_matte_job(job_id: str):
    job = container().jobs.get(job_id)
    if job is None:
        raise HTTPException(404, "job not found")
    return job.to_dict()


@router.post("/recompose")
async def recompose(
    session_id: str = Form(...),
    edited_image: UploadFile = File(...),
):
    try:
        return container().sessions.recompose(session_id, read_upload(edited_image))
    except Exception as exc:
        raise to_http(exc) from exc


@router.get("/image/{session_id}/{name}")
async def get_image(session_id: str, name: str):
    if not name.replace("_", "").isalnum():
        raise HTTPException(400, "bad layer name")
    store = container().store
    try:
        store.require(session_id)
    except Exception as exc:
        raise to_http(exc) from exc
    if not store.has_layer(session_id, name):
        raise HTTPException(404, "layer not found")
    return FileResponse(store.layer_path(session_id, name), media_type="image/png")
