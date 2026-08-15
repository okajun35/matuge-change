"""Lash extraction domain: landmarks, ROI, alignment, evidence, matting."""

from backend.lash_extraction.alignment import align_b_to_a, ecc_refine
from backend.lash_extraction.evidence import (
    darkness_map,
    difference_map,
    eye_prior,
    initial_probability,
)
from backend.lash_extraction.landmarks import (
    ALIGN_POINTS,
    LEFT_EYE,
    RIGHT_EYE,
    detect_landmarks,
    get_landmarker,
)
from backend.lash_extraction.matting import (
    build_trimap,
    composite,
    recompose_onto,
    reconstruction_error,
    run_matting,
)
from backend.lash_extraction.product import product_bbox
from backend.lash_extraction.roi import (
    MAX_ROI_WIDTH,
    MIN_ROI_SIDE,
    EyeRoi,
    compute_eye_roi,
    crop_roi,
    manual_eye_roi,
)

__all__ = [
    "ALIGN_POINTS",
    "LEFT_EYE",
    "MAX_ROI_WIDTH",
    "MIN_ROI_SIDE",
    "RIGHT_EYE",
    "EyeRoi",
    "align_b_to_a",
    "build_trimap",
    "composite",
    "compute_eye_roi",
    "crop_roi",
    "darkness_map",
    "detect_landmarks",
    "difference_map",
    "ecc_refine",
    "eye_prior",
    "get_landmarker",
    "initial_probability",
    "manual_eye_roi",
    "product_bbox",
    "recompose_onto",
    "reconstruction_error",
    "run_matting",
]
