"""Supabase adapters for the catalog, job and stroke ports.

Everything here is optional: without SUPABASE_URL / SUPABASE_SERVICE_ROLE_KEY the
app falls back to the local adapters and keeps running fully offline.
"""

from __future__ import annotations

import contextlib
import os
from datetime import UTC
from functools import lru_cache
from typing import Any

BUCKET = "product-assets"
ASSET_LIST_COLUMNS = (
    "id,name,brand,session_id,storage_path,width,height,alpha_coverage,recon_error,created_at"
)


def is_configured() -> bool:
    return bool(os.environ.get("SUPABASE_URL") and os.environ.get("SUPABASE_SERVICE_ROLE_KEY"))


def public_config() -> dict[str, Any]:
    """Browser-safe config: never exposes the service role key."""
    url = os.environ.get("SUPABASE_URL", "")
    key = os.environ.get("SUPABASE_PUBLISHABLE_KEY", "")
    return {
        "enabled": is_configured(),
        "url": url,
        "publishable_key": key,
        "realtime": bool(is_configured() and url and key),
    }


@lru_cache(maxsize=1)
def client():
    from supabase import create_client

    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
    if not url or not key:
        raise RuntimeError("SUPABASE_URL / SUPABASE_SERVICE_ROLE_KEY are not set")
    return create_client(url, key)


class SupabaseAssetStorage:
    def __init__(self, bucket: str = BUCKET) -> None:
        self._bucket = bucket

    def upload(self, path: str, data: bytes) -> str:
        client().storage.from_(self._bucket).upload(
            path, data, {"content-type": "image/png", "upsert": "true"}
        )
        return path

    def download(self, path: str) -> bytes:
        return client().storage.from_(self._bucket).download(path)


class SupabaseAssetRepository:
    def insert(self, record: dict[str, Any]) -> dict[str, Any]:
        return client().table("product_assets").insert(record).execute().data[0]

    def list(self, limit: int = 50) -> list[dict[str, Any]]:
        return (
            client()
            .table("product_assets")
            .select(ASSET_LIST_COLUMNS)
            .order("created_at", desc=True)
            .limit(limit)
            .execute()
            .data
        )

    def get(self, asset_id: str) -> dict[str, Any] | None:
        rows = client().table("product_assets").select("*").eq("id", asset_id).limit(1).execute().data
        return rows[0] if rows else None

    def similar(self, embedding: list[float], limit: int, exclude_id: str | None) -> list[dict[str, Any]]:
        return (
            client()
            .rpc(
                "match_product_assets",
                {
                    "query_embedding": embedding,
                    "match_count": limit,
                    "exclude_id": exclude_id,
                },
            )
            .execute()
            .data
        )


class SupabaseStrokeRepository:
    def save(self, session_id: str, width: int, height: int, payload: list[dict[str, Any]]) -> None:
        from datetime import datetime

        client().table("session_strokes").upsert(
            {
                "session_id": session_id,
                "width": width,
                "height": height,
                "strokes": payload,
                "updated_at": datetime.now(UTC).isoformat(),
            }
        ).execute()

    def load(self, session_id: str) -> dict[str, Any] | None:
        rows = (
            client().table("session_strokes").select("*").eq("session_id", session_id).limit(1).execute().data
        )
        return rows[0] if rows else None


class SupabaseJobMirror:
    """Mirrors job state into matte_jobs so the browser can follow it over Realtime."""

    def upsert(self, job) -> None:
        row = job.to_dict()
        row.pop("created_at", None)
        # progress mirroring must never break the matting run
        with contextlib.suppress(Exception):
            client().table("matte_jobs").upsert(row).execute()
