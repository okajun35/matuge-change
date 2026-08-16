"""Face landmark detection (MediaPipe Face Landmarker)."""

from __future__ import annotations

import os

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "models",
    "face_landmarker.task",
)

LEFT_EYE = [263, 249, 390, 373, 374, 380, 381, 382, 362, 466, 388, 387, 386, 385, 384, 398]
RIGHT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 246, 161, 160, 159, 158, 157, 173]
ALIGN_POINTS = LEFT_EYE + RIGHT_EYE + [6, 168, 197, 195, 5, 4, 1, 2, 98, 327]

# MediaPipe はモデル入力サイズまで内部で縮小するため、これ以上大きい入力は精度に寄与せず
# メモリ（12MP なら RGB コピー 36MB + 内部バッファ）と時間だけを食う
DETECT_MAX_SIDE = 1600

_landmarker = None


def get_landmarker() -> vision.FaceLandmarker:
    global _landmarker
    if _landmarker is None:
        base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
        options = vision.FaceLandmarkerOptions(base_options=base_options, num_faces=1)
        _landmarker = vision.FaceLandmarker.create_from_options(options)
    return _landmarker


def detect_landmarks(bgr: np.ndarray) -> np.ndarray | None:
    h, w = bgr.shape[:2]
    scale = min(1.0, DETECT_MAX_SIDE / max(h, w))
    small = bgr if scale == 1.0 else cv2.resize(bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = get_landmarker().detect(mp_image)
    del rgb, mp_image, small
    if not result.face_landmarks:
        return None
    # 正規化座標なので、縮小しても元画像のピクセル系に戻せる
    return np.array([[lm.x * w, lm.y * h] for lm in result.face_landmarks[0]], dtype=np.float64)
