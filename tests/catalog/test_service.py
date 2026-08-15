import numpy as np
import pytest

from backend.catalog.asset import AssetDraft
from backend.catalog.service import CatalogService
from tests.catalog.fakes import FakeAssetRepository, FakeAssetStorage


def product_rgba(width: int = 40, height: int = 20) -> np.ndarray:
    rgba = np.zeros((height, width, 4), np.uint8)
    rgba[..., :3] = 90
    rgba[8:12, 5:35, 3] = 255
    return rgba


def draft(name: str = "ボリュームラッシュ D") -> AssetDraft:
    return AssetDraft(
        name=name,
        brand="matuge",
        session_id="sess1",
        rgba=product_rgba(),
        roi=(10, 20, 50, 40),
        roi_scale=0.5,
        landmarks=np.zeros((478, 2)),
        recon_error=2.5,
    )


@pytest.fixture
def service() -> CatalogService:
    return CatalogService(FakeAssetRepository(), FakeAssetStorage())


class TestRegister:
    def test_stores_png_and_row(self, service):
        asset = service.register(draft())
        assert asset.name == "ボリュームラッシュ D"
        assert service.storage.objects[asset.storage_path]
        assert service.repository.rows[0]["storage_path"] == asset.storage_path

    def test_computes_embedding_and_coverage_from_alpha(self, service):
        service.register(draft())
        row = service.repository.rows[0]
        assert len(row["embedding"]) == 64
        assert 0.0 < row["alpha_coverage"] < 1.0
        assert row["recon_error"] == 2.5

    def test_persists_geometry_needed_for_recomposition(self, service):
        service.register(draft())
        row = service.repository.rows[0]
        assert row["roi"] == [10, 20, 50, 40]
        assert row["roi_scale"] == 0.5
        assert len(row["landmarks"]) == 478

    def test_empty_name_is_rejected(self, service):
        with pytest.raises(ValueError):
            service.register(draft(name="  "))

    def test_rgba_without_alpha_channel_is_rejected(self, service):
        bad = draft()
        bad.rgba = np.zeros((10, 10, 3), np.uint8)
        with pytest.raises(ValueError):
            service.register(bad)


class TestSimilarity:
    def test_similar_excludes_the_query_asset(self, service):
        asset = service.register(draft())
        service.find_similar(asset.id, limit=3)
        call = service.repository.similar_calls[-1]
        assert call["exclude_id"] == asset.id
        assert call["limit"] == 3
        assert len(call["embedding"]) == 64

    def test_similar_for_unknown_asset_raises(self, service):
        with pytest.raises(LookupError):
            service.find_similar("00000000-0000-0000-0000-000000000000")


class TestLoading:
    def test_load_rgba_decodes_the_stored_png(self, service):
        asset = service.register(draft())
        loaded = service.load_rgba(asset.id)
        assert loaded.shape == (20, 40, 4)

    def test_list_returns_a_page_with_the_total(self, service):
        service.register(draft(name="A"))
        service.register(draft(name="B"))
        page = service.list_assets(limit=1)
        assert [a["name"] for a in page["items"]] == ["A"]
        assert page["total"] == 2

    def test_list_forwards_offset_and_query(self, service):
        service.register(draft(name="A"))
        page = service.list_assets(limit=5, offset=3, query="vol")
        assert service.repository.page_calls[-1] == {"limit": 5, "offset": 3, "query": "vol"}
        assert page["items"] == []
