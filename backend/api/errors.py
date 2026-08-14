"""Translation of domain errors into HTTP responses."""

from __future__ import annotations

import cv2
import numpy as np
from fastapi import HTTPException, UploadFile

from backend.sessions.errors import (
    FaceNotDetected,
    ImageDecodeError,
    MatteNotReady,
    SessionNotFound,
)

STATUS_BY_ERROR = (
    (SessionNotFound, 404),
    (MatteNotReady, 409),
    (FaceNotDetected, 422),
    (ImageDecodeError, 400),
    (LookupError, 404),
    (ValueError, 400),
)


def to_http(error: Exception) -> HTTPException:
    for error_type, status in STATUS_BY_ERROR:
        if isinstance(error, error_type):
            return HTTPException(status, str(error))
    raise error


def read_upload(file: UploadFile) -> np.ndarray:
    img = cv2.imdecode(np.frombuffer(file.file.read(), np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(400, f"could not decode image: {file.filename}")
    return img
