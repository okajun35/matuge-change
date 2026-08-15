"""Domain errors of the extraction workflow; the API layer maps them to status codes."""

from __future__ import annotations


class SessionNotFound(LookupError):
    def __init__(self, session_id: str) -> None:
        super().__init__(f"session {session_id} not found")


class ImageDecodeError(ValueError):
    pass


class FaceNotDetected(ValueError):
    pass


class MatteNotReady(RuntimeError):
    pass
