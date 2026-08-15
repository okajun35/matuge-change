"""Translation of domain errors into HTTP responses."""

from __future__ import annotations

import cv2
import numpy as np
from fastapi import HTTPException, UploadFile

from backend.sessions.archive import ArchiveUnavailable
from backend.sessions.errors import (
    FaceNotDetected,
    ImageDecodeError,
    MatteNotReady,
    SessionNotFound,
)

STATUS_BY_ERROR = (
    (SessionNotFound, 404),
    (ArchiveUnavailable, 503),
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
    buf = np.frombuffer(file.file.read(), np.uint8)
    # 空バッファは cv2.imdecode が例外を投げる（同期途中・読み取り失敗のファイル）
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR) if buf.size else None
    if img is None:
        raise HTTPException(400, f"could not decode image: {file.filename}")
    return img
