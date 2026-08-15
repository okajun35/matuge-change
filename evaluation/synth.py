"""Procedural lash products and eye-region backgrounds with exact ground truth.

Why procedural instead of a cut-out product photo: the alpha of a cut-out PNG is
only as good as whoever traced it, and the traced edge is usually hard. That would
remove the semi-transparent lash tips, which are exactly what this benchmark needs
to measure. Here a product is a set of vector strands, so

* geometric variation transforms *geometry*, never pixels — ground truth alpha has
  no resampling error at any scale or rotation,
* alpha comes from supersampled coverage, so tips are genuinely fractional,
* the subject's own lashes are generated too, so we know which pixels belong to the
  product and which belong to the person (see `EyeBackground.own_lash_alpha`).

Real product PNGs are still supported by the generator; they are just a second,
less controlled source (`evaluation/products.py`).
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import cv2
import numpy as np

SUPERSAMPLE = 4
LASH_COLOR = (26, 22, 20)  # BGR: near-black with a brown cast


@dataclass(frozen=True)
class Strand:
    """One lash hair (or the lash band) as a polyline that tapers towards its tip."""

    points: np.ndarray  # (n, 2) float32 polyline in output pixels
    root_thickness: float
    color: tuple[int, int, int] = LASH_COLOR
    tip_thickness_ratio: float = 0.25
    root_alpha: float = 1.0
    tip_alpha: float = 0.12


@dataclass(frozen=True)
class LashGeometry:
    strands: tuple[Strand, ...]

    def transformed(self, matrix: np.ndarray) -> LashGeometry:
        """Apply a 2x3 affine to every strand (thickness follows the linear part)."""
        thickness_scale = float(np.sqrt(abs(np.linalg.det(matrix[:, :2]))))
        strands = tuple(
            replace(
                strand,
                points=(strand.points @ matrix[:, :2].T + matrix[:, 2]).astype(np.float32),
                root_thickness=strand.root_thickness * thickness_scale,
            )
            for strand in self.strands
        )
        return LashGeometry(strands)


@dataclass(frozen=True)
class Placement:
    """Similarity placement of a product: the geometry axes of the benchmark."""

    scale: float = 1.0
    rotation_deg: float = 0.0
    offset_x: float = 0.0
    offset_y: float = 0.0
    flip: bool = False


@dataclass(frozen=True)
class EyeBackground:
    """A bare (product-free) eye region plus everything the generator needs to know."""

    image: np.ndarray  # BGR uint8
    own_lash_alpha: np.ndarray  # float32 0..1, the *person's* lashes (zeros if unknown)
    lash_lines: tuple[np.ndarray, ...]  # upper-lid polylines, one per visible eye
    eye_box: tuple[int, int, int, int]
    name: str = "procedural"
    landmarks: np.ndarray | None = None  # MediaPipe landmarks, when the source had a face
    roi_rect: tuple[int, int, int, int] | None = None  # manual ROI for faceless sources

    @property
    def lash_line(self) -> np.ndarray:
        return self.lash_lines[0]

    @property
    def shape(self) -> tuple[int, int]:
        return self.image.shape[:2]

    @property
    def centre(self) -> tuple[float, float]:
        height, width = self.shape
        return (width / 2, height / 2)


def placement_matrix(placement: Placement, centre: tuple[float, float]) -> np.ndarray:
    """2x3 affine for a placement, applied around `centre`."""
    cx, cy = centre
    matrix = cv2.getRotationMatrix2D((0.0, 0.0), placement.rotation_deg, placement.scale)
    if placement.flip:
        matrix[:, 0] *= -1
    matrix[0, 2] += cx + placement.offset_x
    matrix[1, 2] += cy + placement.offset_y
    shift = np.array([[1.0, 0.0, -cx], [0.0, 1.0, -cy], [0.0, 0.0, 1.0]])
    return (np.vstack([matrix, [0, 0, 1]]) @ shift)[:2]


def place(geometry: LashGeometry, placement: Placement, centre: tuple[float, float]) -> LashGeometry:
    return geometry.transformed(placement_matrix(placement, centre))


def _draw_segment(
    alpha: np.ndarray,
    color_buffer: np.ndarray,
    p0: np.ndarray,
    p1: np.ndarray,
    thickness: int,
    value: float,
    color: tuple[int, int, int],
) -> None:
    """Paint one polyline segment, keeping the strongest coverage where strands overlap."""
    height, width = alpha.shape
    pad = thickness + 2
    x0 = int(max(0, min(p0[0], p1[0]) - pad))
    x1 = int(min(width, max(p0[0], p1[0]) + pad))
    y0 = int(max(0, min(p0[1], p1[1]) - pad))
    y1 = int(min(height, max(p0[1], p1[1]) + pad))
    if x0 >= x1 or y0 >= y1:
        return
    patch = np.zeros((y1 - y0, x1 - x0), np.uint8)
    a = (int(round(p0[0])) - x0, int(round(p0[1])) - y0)
    b = (int(round(p1[0])) - x0, int(round(p1[1])) - y0)
    cv2.line(patch, a, b, 255, thickness)
    hit = patch > 0
    region = alpha[y0:y1, x0:x1]
    np.maximum(region, np.where(hit, np.float32(value), np.float32(0.0)), out=region)
    # a basic slice is a view, so writing through the mask reaches `color_buffer`.
    # Bound it to a name anyway: chained `buffer[slice][mask] = x` reads like the
    # copy-then-discard trap it would be if the first index were fancy indexing.
    color_region = color_buffer[y0:y1, x0:x1]
    color_region[hit] = color


def render_geometry(
    geometry: LashGeometry,
    shape: tuple[int, int],
    supersample: int = SUPERSAMPLE,
) -> tuple[np.ndarray, np.ndarray]:
    """Rasterise strands into (BGR uint8, alpha float32 0..1) by supersampled coverage."""
    height, width = shape
    hi = (height * supersample, width * supersample)
    alpha_hi = np.zeros(hi, np.float32)
    color_hi = np.zeros((*hi, 3), np.float32)
    for strand in geometry.strands:
        points = strand.points.astype(np.float64) * supersample
        segments = len(points) - 1
        if segments < 1:
            continue
        for i in range(segments):
            far, near = i / segments, (i + 1) / segments
            taper = 1.0 + (strand.tip_thickness_ratio - 1.0) * near
            thickness = max(1, int(round(strand.root_thickness * supersample * taper)))
            value = strand.root_alpha + (strand.tip_alpha - strand.root_alpha) * far
            _draw_segment(alpha_hi, color_hi, points[i], points[i + 1], thickness, value, strand.color)

    alpha = cv2.resize(alpha_hi, (width, height), interpolation=cv2.INTER_AREA)
    premultiplied = cv2.resize(color_hi * alpha_hi[..., None], (width, height), interpolation=cv2.INTER_AREA)
    safe = np.where(alpha > 1e-6, alpha, 1.0)[..., None]
    bgr = np.clip(premultiplied / safe, 0, 255).astype(np.uint8)
    return bgr, np.clip(alpha, 0.0, 1.0)


def _bezier(p0: np.ndarray, control: np.ndarray, p2: np.ndarray, steps: int = 9) -> np.ndarray:
    t = np.linspace(0.0, 1.0, steps)[:, None]
    return ((1 - t) ** 2 * p0 + 2 * (1 - t) * t * control + t**2 * p2).astype(np.float32)


def _tangents(line: np.ndarray) -> np.ndarray:
    tangent = np.gradient(line, axis=0)
    norm = np.linalg.norm(tangent, axis=1, keepdims=True)
    return tangent / np.maximum(norm, 1e-6)


def synthesize_lash(
    lash_line: np.ndarray,
    seed: int = 0,
    n_strands: int = 40,
    length: float = 0.34,
    curl: float = 0.45,
    thickness: float = 1.6,
    band: bool = True,
    tip_alpha: float = 0.12,
    color: tuple[int, int, int] = LASH_COLOR,
    away_from: np.ndarray | None = None,
) -> LashGeometry:
    """A false-lash product rooted on `lash_line` (inner corner -> outer corner).

    `length` is a fraction of the lash line length; strands fan outwards and get
    longer towards the end of the line, like a real strip lash. `away_from` (normally
    the eye centre) decides which side of the line the strands grow on, so the same
    code works for a left eye, a right eye and a tilted head.
    """
    rng = np.random.default_rng(seed)
    line = np.asarray(lash_line, np.float32)
    span = float(np.linalg.norm(line[-1] - line[0]))
    tangent = _tangents(line)
    normal = np.stack([tangent[:, 1], -tangent[:, 0]], axis=1)
    if away_from is not None:
        outward = line - np.asarray(away_from, np.float32)
        if float(np.sum(normal * outward)) < 0:
            normal = -normal

    strands: list[Strand] = []
    for i in range(n_strands):
        t = (i + 0.5) / n_strands + rng.uniform(-0.4, 0.4) / n_strands
        t = float(np.clip(t, 0.0, 1.0))
        index = t * (len(line) - 1)
        low = int(np.floor(index))
        high = min(low + 1, len(line) - 1)
        frac = index - low
        root = line[low] * (1 - frac) + line[high] * frac
        direction = normal[low] * (1 - frac) + normal[high] * frac
        along = tangent[low] * (1 - frac) + tangent[high] * frac
        # short and upright at the inner corner, long and swept out at the outer corner
        grow = 0.55 + 0.45 * t
        strand_length = span * length * grow * rng.uniform(0.82, 1.18)
        sweep = (0.15 + 0.5 * t) * rng.uniform(0.7, 1.3)
        tip = root + direction * strand_length + along * strand_length * sweep
        control = root + direction * strand_length * 0.55 + along * strand_length * sweep * curl
        jitter = rng.uniform(-0.85, 0.85)
        strands.append(
            Strand(
                points=_bezier(root - direction * span * 0.01, control, tip + jitter),
                root_thickness=thickness * rng.uniform(0.8, 1.25),
                color=tuple(int(np.clip(c + rng.integers(-14, 22), 0, 255)) for c in color),
                tip_alpha=float(tip_alpha * rng.uniform(0.6, 1.4)),
                root_alpha=float(rng.uniform(0.9, 1.0)),
            )
        )
    if band:
        strands.append(
            Strand(
                points=line.astype(np.float32),
                root_thickness=max(1.4, thickness * 1.5),
                color=color,
                tip_thickness_ratio=1.0,
                root_alpha=1.0,
                tip_alpha=1.0,
            )
        )
    return LashGeometry(tuple(strands))


def _skin(width: int, height: int, rng: np.random.Generator) -> np.ndarray:
    """Skin tone with low-frequency shading and film grain."""
    tone = np.array(
        [
            rng.uniform(150, 185),  # B
            rng.uniform(170, 200),  # G
            rng.uniform(195, 225),  # R
        ],
        np.float32,
    )
    # luminance-only shading: independent per-channel noise would look like a rainbow,
    # and the difference map weighs LAB chroma, so fake chroma would flatter it
    coarse = rng.normal(0.0, 1.0, (5, 5)).astype(np.float32)
    shading = cv2.resize(coarse, (width, height), interpolation=cv2.INTER_CUBIC) * 0.05
    grain = rng.normal(0.0, 1.6, (height, width, 1)).astype(np.float32)
    return np.clip(tone * (1.0 + shading[..., None]) + grain, 0, 255)


def _lid_curve(inner: np.ndarray, outer: np.ndarray, bulge: float, steps: int = 28) -> np.ndarray:
    mid = (inner + outer) / 2
    direction = outer - inner
    normal = np.array([direction[1], -direction[0]], np.float32)
    normal /= max(np.linalg.norm(normal), 1e-6)
    return _bezier(inner, mid + normal * bulge, outer, steps)


def synthesize_bare_eye(
    width: int = 320,
    height: int = 240,
    seed: int = 0,
    own_lash: float = 1.0,
    name: str | None = None,
) -> EyeBackground:
    """A product-free eye close-up: skin, lid crease, eye, and the person's own lashes."""
    rng = np.random.default_rng(seed)
    canvas = _skin(width, height, rng)

    inner = np.array([width * rng.uniform(0.13, 0.19), height * rng.uniform(0.58, 0.66)], np.float32)
    outer = np.array([width * rng.uniform(0.82, 0.88), height * rng.uniform(0.44, 0.52)], np.float32)
    span = float(np.linalg.norm(outer - inner))
    upper = _lid_curve(inner, outer, bulge=span * rng.uniform(0.17, 0.22))
    lower = _lid_curve(inner, outer, bulge=-span * rng.uniform(0.09, 0.13))

    # lid crease and socket shading, blurred so the eye sits in a face-like gradient
    shading = np.zeros((height, width), np.float32)
    crease = _lid_curve(inner, outer, bulge=span * rng.uniform(0.34, 0.42))
    cv2.polylines(shading, [crease.astype(np.int32)], False, 1.0, max(2, int(span * 0.035)))
    cv2.polylines(shading, [upper.astype(np.int32)], False, 0.55, max(2, int(span * 0.06)))
    shading = cv2.GaussianBlur(shading, (0, 0), span * 0.035)
    canvas *= (1.0 - 0.30 * np.clip(shading, 0, 1))[..., None]

    eye_polygon = np.vstack([upper, lower[::-1]]).astype(np.int32)
    eye_mask = np.zeros((height, width), np.uint8)
    cv2.fillPoly(eye_mask, [eye_polygon], 255)
    sclera = np.full((height, width, 3), rng.uniform(198, 228), np.float32)
    iris_centre = ((upper[len(upper) // 2] + lower[len(lower) // 2]) / 2).astype(np.int32)
    iris_radius = max(3, int(span * rng.uniform(0.17, 0.21)))
    iris_color = (
        float(rng.uniform(60, 105)),
        float(rng.uniform(55, 95)),
        float(rng.uniform(45, 80)),
    )
    cv2.circle(sclera, tuple(iris_centre), iris_radius, iris_color, -1)
    cv2.circle(sclera, tuple(iris_centre), max(1, int(iris_radius * 0.45)), (18.0, 16.0, 15.0), -1)
    highlight = (iris_centre[0] - iris_radius // 3, iris_centre[1] - iris_radius // 3)
    cv2.circle(sclera, highlight, max(1, iris_radius // 5), (245.0, 245.0, 245.0), -1)
    soft_eye = cv2.GaussianBlur(eye_mask.astype(np.float32) / 255.0, (0, 0), 0.8)[..., None]
    canvas = sclera * soft_eye + canvas * (1 - soft_eye)

    # waterline: the dark rim right at the lash line, present with or without a product
    rim = np.zeros((height, width), np.float32)
    cv2.polylines(rim, [upper.astype(np.int32)], False, 1.0, max(1, int(span * 0.018)))
    rim = cv2.GaussianBlur(rim, (0, 0), max(0.6, span * 0.006))
    canvas *= (1.0 - 0.55 * np.clip(rim, 0, 1))[..., None]

    own_alpha = np.zeros((height, width), np.float32)
    if own_lash > 0:
        own = synthesize_lash(
            upper,
            seed=seed + 977,
            n_strands=26,
            length=0.13,
            curl=0.3,
            thickness=1.0,
            band=False,
            tip_alpha=0.05,
            color=(38, 34, 32),
        )
        own_bgr, own_alpha = render_geometry(own, (height, width))
        own_alpha = np.clip(own_alpha * own_lash, 0.0, 1.0)
        canvas = own_bgr.astype(np.float32) * own_alpha[..., None] + canvas * (1 - own_alpha[..., None])

    box = np.vstack([upper, lower]).astype(np.int32)
    eye_box = (
        int(max(0, box[:, 0].min())),
        int(max(0, box[:, 1].min())),
        int(min(width, box[:, 0].max() + 1)),
        int(min(height, box[:, 1].max() + 1)),
    )
    return EyeBackground(
        image=np.clip(canvas, 0, 255).astype(np.uint8),
        own_lash_alpha=own_alpha,
        lash_lines=(upper,),
        eye_box=eye_box,
        name=name or f"procedural_{seed:04d}",
        roi_rect=None,  # filled in by evaluation.backgrounds with the fixed ROI rule
    )


def synthesize_product(
    background: EyeBackground,
    seed: int = 0,
    **params: object,
) -> LashGeometry:
    """One product covering every visible eye of a background (strands are unioned)."""
    strands: list[Strand] = []
    for index, line in enumerate(background.lash_lines):
        centre = np.asarray(line).mean(axis=0)
        # grow away from the eye: use the eye box centre, which is inside the eye
        x0, y0, x1, y1 = background.eye_box
        inside = np.array([centre[0], (y0 + y1) / 2], np.float32)
        geometry = synthesize_lash(line, seed=seed + index * 31, away_from=inside, **params)  # type: ignore[arg-type]
        strands.extend(geometry.strands)
    return LashGeometry(tuple(strands))
