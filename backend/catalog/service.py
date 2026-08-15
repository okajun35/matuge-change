"""Use cases for the product lash catalog."""

from __future__ import annotations

import uuid
from typing import Any, Protocol

import cv2
import numpy as np

from backend.catalog.asset import AssetDraft, ProductAsset
from backend.catalog.descriptor import LashDescriptor, alpha_coverage


class AssetStorage(Protocol):
    def upload(self, path: str, data: bytes) -> str: ...

    def download(self, path: str) -> bytes: ...


class ProductAssetRepository(Protocol):
    def insert(self, record: dict[str, Any]) -> dict[str, Any]: ...

    def page(self, limit: int = 12, offset: int = 0, query: str = "") -> dict[str, Any]: ...

    def get(self, asset_id: str) -> dict[str, Any] | None: ...

    def similar(self, embedding: list[float], limit: int, exclude_id: str | None) -> list[dict[str, Any]]: ...


class CatalogService:
    def __init__(self, repository: ProductAssetRepository, storage: AssetStorage) -> None:
        self.repository = repository
        self.storage = storage

    def register(self, draft: AssetDraft) -> ProductAsset:
        draft.validate()
        alpha = draft.alpha
        height, width = alpha.shape
        path = f"{draft.session_id or 'adhoc'}/{uuid.uuid4()}.png"
        self.storage.upload(path, cv2.imencode(".png", draft.rgba)[1].tobytes())

        row = self.repository.insert(
            {
                "name": draft.name.strip(),
                "brand": draft.brand,
                "session_id": draft.session_id,
                "storage_path": path,
                "roi": [int(v) for v in draft.roi],
                "roi_scale": float(draft.roi_scale),
                "landmarks": np.asarray(draft.landmarks).tolist(),
                "width": int(width),
                "height": int(height),
                "alpha_coverage": alpha_coverage(alpha),
                "recon_error": draft.recon_error,
                "embedding": LashDescriptor.from_alpha(alpha).to_list(),
            }
        )
        return ProductAsset.from_row(row)

    def list_assets(self, limit: int = 12, offset: int = 0, query: str = "") -> dict[str, Any]:
        """One page of the catalog: `{"items": [...], "total": n}`."""
        return self.repository.page(limit=limit, offset=offset, query=query)

    def get_asset(self, asset_id: str) -> dict[str, Any]:
        return self._require(asset_id)

    def find_similar(self, asset_id: str, limit: int = 5) -> list[dict[str, Any]]:
        row = self._require(asset_id)
        embedding = row.get("embedding")
        if embedding is None:
            raise LookupError(f"asset {asset_id} has no embedding")
        return self.repository.similar(list(embedding), limit=limit, exclude_id=asset_id)

    def load_rgba(self, asset_id: str) -> np.ndarray:
        row = self._require(asset_id)
        return self.decode_png(self.storage.download(row["storage_path"]))

    def load_png(self, asset_id: str) -> bytes:
        return self.storage.download(self._require(asset_id)["storage_path"])

    def load_mask_png(self, asset_id: str) -> bytes:
        """Alpha channel of the stored product PNG, as a grayscale PNG."""
        return cv2.imencode(".png", self.load_rgba(asset_id)[..., 3])[1].tobytes()

    @staticmethod
    def decode_png(data: bytes) -> np.ndarray:
        return cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_UNCHANGED)

    def _require(self, asset_id: str) -> dict[str, Any]:
        row = self.repository.get(asset_id)
        if row is None:
            raise LookupError(f"asset {asset_id} not found")
        return row
