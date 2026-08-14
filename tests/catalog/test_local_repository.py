import numpy as np
import pytest

from backend.catalog.local import LocalAssetRepository, LocalAssetStorage


@pytest.fixture
def repository(tmp_path) -> LocalAssetRepository:
    return LocalAssetRepository(str(tmp_path / "assets"))


def record(name: str, embedding: list[float]) -> dict:
    return {
        "name": name,
        "brand": None,
        "session_id": "s",
        "storage_path": f"{name}.png",
        "roi": [0, 0, 10, 10],
        "roi_scale": 1.0,
        "landmarks": [],
        "width": 10,
        "height": 10,
        "alpha_coverage": 0.2,
        "recon_error": 1.0,
        "embedding": embedding,
    }


class TestLocalAssetRepository:
    def test_insert_assigns_id_and_timestamp(self, repository):
        row = repository.insert(record("a", [1.0, 0.0]))
        assert row["id"] and row["created_at"]

    def test_rows_survive_a_new_repository_instance(self, repository, tmp_path):
        repository.insert(record("a", [1.0, 0.0]))
        reopened = LocalAssetRepository(str(tmp_path / "assets"))
        assert [r["name"] for r in reopened.list()] == ["a"]

    def test_get_returns_none_for_unknown_id(self, repository):
        assert repository.get("missing") is None

    def test_similar_orders_by_cosine_similarity(self, repository):
        repository.insert(record("near", [1.0, 0.1]))
        repository.insert(record("far", [0.0, 1.0]))
        results = repository.similar([1.0, 0.0], limit=2, exclude_id=None)
        assert [r["name"] for r in results] == ["near", "far"]
        assert results[0]["similarity"] > results[1]["similarity"]

    def test_similar_excludes_the_given_id(self, repository):
        row = repository.insert(record("self", [1.0, 0.0]))
        repository.insert(record("other", [0.9, 0.2]))
        results = repository.similar([1.0, 0.0], limit=5, exclude_id=row["id"])
        assert [r["name"] for r in results] == ["other"]


class TestLocalAssetStorage:
    def test_upload_then_download_roundtrip(self, tmp_path):
        storage = LocalAssetStorage(str(tmp_path))
        storage.upload("sess/one.png", b"\x89PNG-data")
        assert storage.download("sess/one.png") == b"\x89PNG-data"

    def test_download_missing_object_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            LocalAssetStorage(str(tmp_path)).download("nope.png")

    def test_rejects_paths_escaping_the_root(self, tmp_path):
        with pytest.raises(ValueError):
            LocalAssetStorage(str(tmp_path)).upload("../evil.png", np.zeros(1, np.uint8).tobytes())
