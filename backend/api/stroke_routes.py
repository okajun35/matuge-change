"""Brush stroke persistence endpoints (resume work on a session)."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter
from pydantic import BaseModel

from backend.api.container import container
from backend.api.errors import to_http
from backend.strokes.stroke import StrokeSet

router = APIRouter(prefix="/api/sessions")


class StrokePayload(BaseModel):
    width: int
    height: int
    strokes: list[dict[str, Any]]


@router.put("/{session_id}/strokes")
async def save_strokes(session_id: str, payload: StrokePayload):
    app = container()
    try:
        app.store.require(session_id)
        strokes = StrokeSet.from_payload(payload.width, payload.height, payload.strokes)
        app.strokes.save(session_id, strokes)
    except Exception as exc:
        raise to_http(exc) from exc
    return {"saved": len(strokes)}


@router.get("/{session_id}/strokes")
async def load_strokes(session_id: str):
    app = container()
    try:
        app.store.require(session_id)
        saved = app.strokes.load(session_id)
    except Exception as exc:
        raise to_http(exc) from exc
    if saved is None:
        meta = app.store.load_meta(session_id)
        return {"width": meta["width"], "height": meta["height"], "strokes": []}
    return {"width": saved.width, "height": saved.height, "strokes": saved.to_payload()}
