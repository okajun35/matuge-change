"""Composition root: picks Supabase adapters when configured, local ones otherwise."""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache

from backend.catalog.local import LocalAssetRepository, LocalAssetStorage
from backend.catalog.service import CatalogService
from backend.infrastructure import supabase_gateway
from backend.jobs.repository import InMemoryJobRepository, MirroringJobRepository
from backend.jobs.runner import MatteJobRunner
from backend.sessions.service import SessionService
from backend.sessions.store import SessionStore
from backend.strokes.repository import FileStrokeRepository
from backend.strokes.service import StrokeService

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(ROOT, "data")
FRONTEND_DIR = os.path.join(ROOT, "frontend")


@dataclass(frozen=True)
class Container:
    sessions: SessionService
    catalog: CatalogService
    strokes: StrokeService
    jobs: MatteJobRunner

    @property
    def store(self) -> SessionStore:
        return self.sessions.store


@lru_cache(maxsize=1)
def container() -> Container:
    store = SessionStore(DATA_DIR)
    supabase = supabase_gateway.is_configured()

    if supabase:
        catalog = CatalogService(
            supabase_gateway.SupabaseAssetRepository(), supabase_gateway.SupabaseAssetStorage()
        )
        strokes = StrokeService(supabase_gateway.SupabaseStrokeRepository())
        jobs = MirroringJobRepository(InMemoryJobRepository(), supabase_gateway.SupabaseJobMirror())
    else:
        assets_dir = os.path.join(DATA_DIR, "assets")
        catalog = CatalogService(LocalAssetRepository(assets_dir), LocalAssetStorage(assets_dir))
        strokes = StrokeService(FileStrokeRepository(DATA_DIR))
        jobs = InMemoryJobRepository()

    return Container(
        sessions=SessionService(store),
        catalog=catalog,
        strokes=strokes,
        jobs=MatteJobRunner(jobs),
    )
