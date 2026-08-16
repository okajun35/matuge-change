"""FastAPI app for the lash extraction PoC."""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from backend.api import (
    catalog_routes,
    config_routes,
    sessions_routes,
    stroke_routes,
    video_routes,
)
from backend.api.container import DATA_DIR, FRONTEND_DIR, container
from backend.observability import log_matte_settings

log_matte_settings()  # which solve mode this deployment runs, before anything serves

app = FastAPI(title="matuge-change PoC")
app.include_router(config_routes.router)
app.include_router(sessions_routes.router)
app.include_router(catalog_routes.router)
app.include_router(stroke_routes.router)
app.include_router(video_routes.router)

container()  # fail fast on misconfiguration and create the data directory

app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")

__all__ = ["DATA_DIR", "FRONTEND_DIR", "app"]
