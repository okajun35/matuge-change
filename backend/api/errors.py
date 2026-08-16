"""Translation of domain errors into HTTP responses."""

from __future__ import annotations

from collections.abc import Callable

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
    return decode_upload(file)()


def decode_upload(file: UploadFile) -> Callable[[], np.ndarray]:
    """Defer decoding so the caller can decode one image at a time.

    A phone photo is ~2MB compressed but ~36MB as an array; keeping the bytes and
    decoding on demand is what lets a small host handle two uploads.
    """
    data = file.file.read()
    filename = file.filename

    def decode() -> np.ndarray:
        # 空バッファは cv2.imdecode が例外を投げる（同期途中・読み取り失敗のファイル）
        buf = np.frombuffer(data, np.uint8)
        img = cv2.imdecode(buf, cv2.IMREAD_COLOR) if buf.size else None
        if img is None:
            raise HTTPException(400, f"could not decode image: {filename}")
        return img

    return decode
