"""In-memory doubles for the catalog ports, so the service is testable offline."""

from __future__ import annotations

import uuid
from typing import Any


class FakeAssetStorage:
    def __init__(self) -> None:
        self.objects: dict[str, bytes] = {}

    def upload(self, path: str, data: bytes) -> str:
        self.objects[path] = data
        return path

    def download(self, path: str) -> bytes:
        return self.objects[path]


class FakeAssetRepository:
    def __init__(self) -> None:
        self.rows: list[dict[str, Any]] = []
        self.similar_calls: list[dict[str, Any]] = []
        self.page_calls: list[dict[str, Any]] = []

    def insert(self, record: dict[str, Any]) -> dict[str, Any]:
        row = {**record, "id": str(uuid.uuid4()), "created_at": "2026-01-01T00:00:00Z"}
        self.rows.append(row)
        return row

    def page(self, limit: int = 12, offset: int = 0, query: str = "") -> dict[str, Any]:
        self.page_calls.append({"limit": limit, "offset": offset, "query": query})
        rows = [r for r in self.rows if query.lower() in r["name"].lower()]
        return {"items": rows[offset : offset + limit], "total": len(rows)}

    def get(self, asset_id: str) -> dict[str, Any] | None:
        return next((r for r in self.rows if r["id"] == asset_id), None)

    def similar(self, embedding: list[float], limit: int, exclude_id: str | None):
        self.similar_calls.append({"embedding": embedding, "limit": limit, "exclude_id": exclude_id})
        return [r for r in self.rows if r["id"] != exclude_id][:limit]
