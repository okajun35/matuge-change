"""Use cases for resuming brush work on a session."""

from __future__ import annotations

from backend.strokes.repository import StrokeRepository
from backend.strokes.stroke import StrokeSet


class StrokeService:
    def __init__(self, repository: StrokeRepository) -> None:
        self._repository = repository

    def save(self, session_id: str, strokes: StrokeSet) -> None:
        self._repository.save(session_id, strokes.width, strokes.height, strokes.to_payload())

    def load(self, session_id: str) -> StrokeSet | None:
        record = self._repository.load(session_id)
        if record is None:
            return None
        return StrokeSet.from_payload(record["width"], record["height"], record["strokes"])
