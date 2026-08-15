import pytest

from backend.strokes.repository import FileStrokeRepository
from backend.strokes.service import StrokeService
from backend.strokes.stroke import StrokeSet

PAYLOAD = [{"tool": "fg", "radius": 5, "points": [[1, 1], [2, 2]]}]


@pytest.fixture
def service(tmp_path) -> StrokeService:
    return StrokeService(FileStrokeRepository(str(tmp_path)))


class TestStrokeService:
    def test_save_then_load_roundtrip(self, service):
        service.save("sess1", StrokeSet.from_payload(64, 32, PAYLOAD))
        loaded = service.load("sess1")
        assert loaded.width == 64 and loaded.height == 32
        assert loaded.to_payload() == PAYLOAD

    def test_load_unknown_session_returns_none(self, service):
        assert service.load("nope") is None

    def test_save_replaces_previous_strokes(self, service):
        service.save("sess1", StrokeSet.from_payload(64, 32, PAYLOAD))
        service.save("sess1", StrokeSet.from_payload(64, 32, []))
        assert service.load("sess1").to_payload() == []

    def test_constraints_are_rebuilt_from_saved_strokes(self, service):
        service.save(
            "sess1",
            StrokeSet.from_payload(
                40,
                40,
                [
                    {"tool": "bg", "radius": 4, "points": [[10, 10]]},
                ],
            ),
        )
        assert service.load("sess1").rasterize()[10, 10] == -1
