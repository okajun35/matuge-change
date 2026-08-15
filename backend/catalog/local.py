"""Filesystem-backed catalog adapters, so the PoC works without Supabase."""

from __future__ import annotations

import json
import os
import uuid
from datetime import UTC, datetime
from typing import Any

import numpy as np

LIST_FIELDS = (
    "id",
    "name",
    "brand",
    "session_id",
    "storage_path",
    "width",
    "height",
    "alpha_coverage",
    "recon_error",
    "created_at",
)


class LocalAssetStorage:
    def __init__(self, root: str) -> None:
        self._root = os.path.abspath(root)
        os.makedirs(self._root, exist_ok=True)

    def _resolve(self, path: str) -> str:
        full = os.path.abspath(os.path.join(self._root, path))
        if os.path.commonpath([self._root, full]) != self._root:
            raise ValueError(f"path escapes the storage root: {path!r}")
        return full

    def upload(self, path: str, data: bytes) -> str:
        full = self._resolve(path)
        os.makedirs(os.path.dirname(full), exist_ok=True)
        with open(full, "wb") as f:
            f.write(data)
        return path

    def download(self, path: str) -> bytes:
        with open(self._resolve(path), "rb") as f:
            return f.read()


class LocalAssetRepository:
    """A JSON index with brute-force cosine search — the pgvector stand-in."""

    def __init__(self, root: str) -> None:
        os.makedirs(root, exist_ok=True)
        self._index = os.path.join(root, "index.json")

    def _load(self) -> list[dict[str, Any]]:
        if not os.path.exists(self._index):
            return []
        with open(self._index) as f:
            return json.load(f)

    def _store(self, rows: list[dict[str, Any]]) -> None:
        with open(self._index, "w") as f:
            json.dump(rows, f)

    def insert(self, record: dict[str, Any]) -> dict[str, Any]:
        rows = self._load()
        row = {
            **record,
            "id": str(uuid.uuid4()),
            "created_at": datetime.now(UTC).isoformat(),
        }
        rows.append(row)
        self._store(rows)
        return row

    def page(self, limit: int = 12, offset: int = 0, query: str = "") -> dict[str, Any]:
        rows = list(enumerate(self._load()))
        if query:
            needle = query.casefold()
            rows = [
                pair
                for pair in rows
                if needle in f"{pair[1].get('name') or ''} {pair[1].get('brand') or ''}".casefold()
            ]
        rows.sort(key=lambda pair: (pair[1]["created_at"], pair[0]), reverse=True)
        window = rows[offset : offset + limit]
        return {
            "items": [{k: row.get(k) for k in LIST_FIELDS} for _, row in window],
            "total": len(rows),
        }

    def get(self, asset_id: str) -> dict[str, Any] | None:
        return next((r for r in self._load() if r["id"] == asset_id), None)

    def similar(self, embedding: list[float], limit: int, exclude_id: str | None) -> list[dict[str, Any]]:
        query = np.asarray(embedding, np.float32)
        scored = []
        for row in self._load():
            if row["id"] == exclude_id or not row.get("embedding"):
                continue
            other = np.asarray(row["embedding"], np.float32)
            denominator = float(np.linalg.norm(query) * np.linalg.norm(other)) + 1e-12
            scored.append((float(query @ other / denominator), row))
        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [
            {k: row.get(k) for k in ("id", "name", "brand", "storage_path")} | {"similarity": score}
            for score, row in scored[:limit]
        ]
