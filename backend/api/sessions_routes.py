"""Session, matting and recomposition endpoints."""

from __future__ import annotations

from math import isfinite

import numpy as np
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse

from backend.api.container import container
from backend.api.errors import read_upload, to_http
from backend.strokes.constraints import decode_constraints_png

router = APIRouter(prefix="/api")


def _rect(raw: str, name: str) -> tuple[float, float, float, float] | None:
    """Parse a user-drawn rectangle; an empty value means it was not specified."""
    if not raw.strip():
        return None
    parts = raw.split(",")
    if len(parts) != 4:
        raise HTTPException(400, f"{name} must be 'x0,y0,x1,y1'")
    try:
        x0, y0, x1, y1 = (float(p) for p in parts)
    except ValueError as exc:
        raise HTTPException(400, f"{name} must be four numbers") from exc
    if not all(isfinite(v) for v in (x0, y0, x1, y1)):
        raise HTTPException(400, f"{name} must be finite numbers")
    return x0, y0, x1, y1


def _roi_rect(raw: str) -> tuple[float, float, float, float] | None:
    return _rect(raw, "roi_rect")


@router.post("/session")
async def create_session(
    image_with: UploadFile = File(...),
    image_without: UploadFile | None = File(None),
    roi_rect: str = Form(""),
):
    img_a = read_upload(image_with)
    img_b = read_upload(image_without) if image_without is not None else None
    rect = _roi_rect(roi_rect)
    try:
        return container().sessions.create(img_a, img_b, rect)
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
    "source_with",
    "source_without",
    "source_edited",
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
        "mode": meta.get("mode", "auto"),
        "roi_rect": meta.get("roi"),
        "dest_rect": meta.get("dest_rect"),
        "layers": [name for name in LAYER_ORDER if store.has_layer(session_id, name)],
    }


@router.post("/sessions/{session_id}/archive")
async def export_session(session_id: str):
    """セッションのファイル一式を Supabase Storage へ退避する。"""
    try:
        return container().archive.export(session_id)
    except Exception as exc:
        raise to_http(exc) from exc


@router.post("/sessions/{session_id}/archive/restore")
async def restore_session(session_id: str):
    """退避済みセッションをこのマシンの `data/` に復元する。"""
    try:
        return container().archive.restore(session_id)
    except Exception as exc:
        raise to_http(exc) from exc


@router.get("/sessions/{session_id}/runs")
async def get_runs(session_id: str):
    try:
        return {"runs": container().sessions.runs(session_id)}
    except Exception as exc:
        raise to_http(exc) from exc


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
    dest_rect: str = Form(""),
):
    try:
        rect = _rect(dest_rect, "dest_rect")
        return container().sessions.recompose(session_id, read_upload(edited_image), rect)
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
