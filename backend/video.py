"""Video mode: best-frame selection and per-frame eye-region replacement compositing.

The product (lash) pixels are reused as-is from the source frame: one best frame
is AI-edited externally, then each original frame's eye region — lashes, lids and
their natural blink motion — is pasted back over the edited image, which is the
only side that gets warped. Compositing leaves the product pixels untransformed;
the feathered mask border and the output H.264 re-encode are the only losses.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from collections.abc import Callable

import cv2
import numpy as np

from backend.lash_extraction import ALIGN_POINTS, LEFT_EYE, RIGHT_EYE

# Eye-aspect-ratio landmark indices (MediaPipe face mesh)
_R_H = (33, 133)
_R_V = ((159, 145), (158, 153))
_L_H = (362, 263)
_L_V = ((386, 374), (385, 380))

MAX_FRAMES = 600

# Keep temporal filtering from visibly separating the original eye pixels from
# the edited face during deliberate, fast motion.  Values are per frame and in
# the same units as ``similarity_parameters``.
_MAX_POSITION_CORRECTION = 2.0
_MAX_ANGLE_CORRECTION = np.deg2rad(0.5)
_MAX_LOG_SCALE_CORRECTION = np.log(1.005)


def eye_openness(lms: np.ndarray) -> float:
    """Mean eye aspect ratio (vertical opening / horizontal width) over both eyes."""
    ratios = []
    for (h0, h1), verts in ((_R_H, _R_V), (_L_H, _L_V)):
        width = np.linalg.norm(lms[h0] - lms[h1])
        if width < 1e-6:
            continue
        vert = np.mean([np.linalg.norm(lms[t] - lms[b]) for t, b in verts])
        ratios.append(vert / width)
    return float(np.mean(ratios)) if ratios else 0.0


def laplacian_sharpness(bgr: np.ndarray) -> float:
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def select_best_frame(frames: list[np.ndarray], landmarks: list[np.ndarray | None]) -> int:
    """Pick the frame with the sharpest image and most open eyes."""
    valid = [i for i, lms in enumerate(landmarks) if lms is not None]
    if not valid:
        raise ValueError("no face detected in any frame")
    sharp = np.array([laplacian_sharpness(frames[i]) for i in valid])
    sharp = sharp / max(sharp.max(), 1e-6)
    open_ = np.array([eye_openness(landmarks[i]) for i in valid])
    open_ = open_ / max(open_.max(), 1e-6)
    scores = 0.5 * sharp + 0.5 * open_
    return valid[int(np.argmax(scores))]


def eye_region_mask(shape: tuple, lms: np.ndarray, expand: float = 0.45) -> np.ndarray:
    """Feathered [0,1] mask over the eye regions (lashes included) in frame coords.

    ``expand`` is the dilation radius as a fraction of the average eye width.
    """
    h, w = shape[:2]
    mask = np.zeros((h, w), np.uint8)
    widths = []
    for idx_set, (h0, h1) in ((RIGHT_EYE, _R_H), (LEFT_EYE, _L_H)):
        cv2.fillPoly(mask, [lms[idx_set].astype(np.int32)], 255)
        widths.append(np.linalg.norm(lms[h0] - lms[h1]))
    radius = max(2, int(expand * float(np.mean(widths))))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
    dilated = cv2.dilate(mask, kernel)
    feather = max(1.0, radius * 0.35)
    soft = cv2.GaussianBlur(dilated.astype(np.float32) / 255.0, (0, 0), feather)
    return np.clip(soft.astype(np.float64) * 1.2, 0.0, 1.0)


def blend_with_mask(frame: np.ndarray, edited: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """mask=1 keeps the original frame (eye region), mask=0 takes the edited image."""
    m = mask[..., None]
    out = m * frame.astype(np.float64) + (1.0 - m) * edited.astype(np.float64)
    return np.clip(out, 0, 255).astype(np.uint8)


def warp_edited_to_frame(
    edited: np.ndarray,
    lms_edited: np.ndarray,
    lms_frame: np.ndarray,
    frame_shape: tuple,
) -> np.ndarray:
    """Warp the AI-edited image into a frame's coordinates via landmark affine."""
    matrix = estimate_similarity_transform(lms_edited, lms_frame)
    if matrix is None:
        matrix = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float64)
    return warp_edited_with_transform(edited, matrix, frame_shape)


def estimate_similarity_transform(source: np.ndarray, destination: np.ndarray) -> np.ndarray | None:
    """Estimate the similarity transform from source landmarks to destination landmarks."""
    src = source[ALIGN_POINTS].astype(np.float32)
    dst = destination[ALIGN_POINTS].astype(np.float32)
    matrix, _ = cv2.estimateAffinePartial2D(src, dst, method=cv2.LMEDS)
    return matrix


def warp_edited_with_transform(edited: np.ndarray, matrix: np.ndarray, frame_shape: tuple) -> np.ndarray:
    """Warp the AI-edited image with a precomputed similarity transform."""
    h, w = frame_shape[:2]
    return cv2.warpAffine(edited, matrix, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def similarity_parameters(matrix: np.ndarray, anchor: np.ndarray | None = None) -> np.ndarray:
    """Return anchor position, rotation, and log-scale for a similarity matrix."""
    if matrix.shape != (2, 3):
        raise ValueError("similarity matrix must have shape (2, 3)")
    scale = float(np.hypot(matrix[0, 0], matrix[1, 0]))
    if scale <= 1e-8:
        raise ValueError("similarity matrix scale must be positive")
    position = matrix[:, 2] if anchor is None else matrix[:, :2] @ anchor + matrix[:, 2]
    return np.array([*position, np.arctan2(matrix[1, 0], matrix[0, 0]), np.log(scale)])


def similarity_matrix(parameters: np.ndarray, anchor: np.ndarray | None = None) -> np.ndarray:
    """Build a similarity matrix from anchor position, rotation, and log-scale."""
    x, y, angle, log_scale = parameters
    scale = float(np.exp(log_scale))
    cosine = scale * np.cos(angle)
    sine = scale * np.sin(angle)
    linear = np.array([[cosine, -sine], [sine, cosine]], dtype=np.float64)
    position = np.array([x, y], dtype=np.float64)
    translation = position if anchor is None else position - linear @ anchor
    return np.column_stack((linear, translation))


def interpolate_similarity_transforms(
    transforms: list[np.ndarray | None], anchor: np.ndarray | None = None
) -> list[np.ndarray | None]:
    """Fill transform gaps linearly; keep an all-missing series missing."""
    if not transforms:
        return []
    parameters = np.full((len(transforms), 4), np.nan, dtype=np.float64)
    for index, matrix in enumerate(transforms):
        if matrix is not None:
            parameters[index] = similarity_parameters(matrix, anchor)
    valid = np.isfinite(parameters).all(axis=1)
    if not valid.any():
        return [None] * len(transforms)

    valid_index = np.flatnonzero(valid)
    parameters[valid, 2] = np.unwrap(parameters[valid, 2])
    index = np.arange(len(transforms))
    for column in range(parameters.shape[1]):
        parameters[:, column] = np.interp(index, valid_index, parameters[valid, column])
    return [similarity_matrix(row, anchor) for row in parameters]


def _centered_median(values: np.ndarray, radius: int) -> np.ndarray:
    """Suppress one-frame transform outliers without introducing temporal delay."""
    padded = np.pad(values, (radius, radius), mode="edge")
    return np.array([np.median(padded[i : i + 2 * radius + 1]) for i in range(len(values))])


def _centered_mean(values: np.ndarray, radius: int) -> np.ndarray:
    """Smooth a trajectory while keeping a constant-velocity interior unchanged."""
    padded = np.pad(values, (radius, radius), mode="edge")
    kernel = np.full(2 * radius + 1, 1 / (2 * radius + 1))
    return np.convolve(padded, kernel, mode="valid")


def stabilize_similarity_transforms(
    transforms: list[np.ndarray | None], fps: float, anchor: np.ndarray | None = None
) -> list[np.ndarray | None]:
    """Interpolate and temporally smooth the edited-to-frame transform trajectory."""
    filled = interpolate_similarity_transforms(transforms, anchor)
    if not filled or all(matrix is None for matrix in filled):
        return filled

    parameters = np.array([similarity_parameters(matrix, anchor) for matrix in filled if matrix is not None])
    radius = max(1, int(round(fps * 0.05)))
    stabilized = np.empty_like(parameters)
    for column in range(parameters.shape[1]):
        median = _centered_median(parameters[:, column], radius)
        stabilized[:, column] = _centered_mean(median, radius)
    correction = stabilized - parameters
    position_norm = np.linalg.norm(correction[:, :2], axis=1)
    position_factor = np.minimum(1.0, _MAX_POSITION_CORRECTION / np.maximum(position_norm, 1e-12))
    correction[:, :2] *= position_factor[:, None]
    correction[:, 2] = np.clip(correction[:, 2], -_MAX_ANGLE_CORRECTION, _MAX_ANGLE_CORRECTION)
    correction[:, 3] = np.clip(correction[:, 3], -_MAX_LOG_SCALE_CORRECTION, _MAX_LOG_SCALE_CORRECTION)
    stabilized = parameters + correction
    return [similarity_matrix(row, anchor) for row in stabilized]


def transform_landmarks(landmarks: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Move landmark coordinates with a similarity matrix without warping source pixels."""
    return landmarks @ matrix[:, :2].T + matrix[:, 2]


def estimate_similarity_transforms(
    lms_edited: np.ndarray, landmarks: list[np.ndarray | None]
) -> list[np.ndarray | None]:
    """Estimate one edited-to-frame transform for every detected frame."""
    return [None if lms is None else estimate_similarity_transform(lms_edited, lms) for lms in landmarks]


def compose_frames(
    frames: list[np.ndarray],
    landmarks: list[np.ndarray | None],
    edited: np.ndarray,
    lms_edited: np.ndarray,
    expand: float = 0.45,
    fps: float = 30.0,
    progress: Callable[[int, int], None] | None = None,
) -> list[np.ndarray]:
    """Per frame: align edited image to the face, then paste the frame's eye region on top."""
    raw_transforms = estimate_similarity_transforms(lms_edited, landmarks)
    face_anchor = lms_edited[ALIGN_POINTS].mean(axis=0)
    transforms = stabilize_similarity_transforms(raw_transforms, fps=fps, anchor=face_anchor)
    outs: list[np.ndarray] = []
    for index, (frame, matrix) in enumerate(zip(frames, transforms, strict=True), start=1):
        if matrix is None:
            outs.append(frame.copy())
        else:
            warped = warp_edited_with_transform(edited, matrix, frame.shape)
            mask = eye_region_mask(frame.shape, transform_landmarks(lms_edited, matrix), expand)
            outs.append(blend_with_mask(frame, warped, mask))
        if progress is not None:
            progress(index, len(frames))
    return outs


def read_video_frames(path: str, max_frames: int = MAX_FRAMES) -> tuple[list[np.ndarray], float]:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise ValueError(f"could not open video: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frames: list[np.ndarray] = []
    while len(frames) < max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()
    if not frames:
        raise ValueError(f"no frames decoded from video: {path}")
    return frames, float(fps)


def write_video(frames: list[np.ndarray], fps: float, path: str) -> None:
    """Write frames to mp4; re-encode to H.264 via ffmpeg when available (browser playback)."""
    h, w = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    with tempfile.TemporaryDirectory() as tmp:
        raw = os.path.join(tmp, "raw.mp4")
        writer = cv2.VideoWriter(raw, fourcc, fps, (w, h))
        for frame in frames:
            writer.write(frame)
        writer.release()
        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg:
            result = subprocess.run(
                [
                    ffmpeg,
                    "-y",
                    "-i",
                    raw,
                    "-c:v",
                    "libx264",
                    "-pix_fmt",
                    "yuv420p",
                    "-movflags",
                    "+faststart",
                    path,
                ],
                capture_output=True,
            )
            if result.returncode == 0:
                return
        shutil.copyfile(raw, path)
