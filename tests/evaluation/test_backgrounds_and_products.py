"""The photo-folder and real-product paths, which the default run never exercises.

`--background-dir` and `--product` are advertised as supported, so they need cover even
though the shipped benchmark uses procedural data. MediaPipe is stubbed: whether the
detector fires on a given photo is its business, not this module's.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from evaluation import backgrounds
from evaluation.products import ImageProduct, ProceduralProduct, load_product_png
from evaluation.synth import Placement


@pytest.fixture
def photo(tmp_path) -> str:
    directory = tmp_path / "photos"
    directory.mkdir()
    rng = np.random.default_rng(0)
    for name in ("a.jpg", "b.png"):
        image = rng.integers(90, 180, (200, 260, 3), dtype=np.uint8)
        cv2.imwrite(str(directory / name), image)
    (directory / "notes.txt").write_text("ignored")
    return str(directory)


class TestImageBackgrounds:
    def test_skips_photos_without_a_face_by_default(self, photo, monkeypatch):
        monkeypatch.setattr(backgrounds, "detect_landmarks", lambda _image: None)
        assert list(backgrounds.image_backgrounds(photo)) == []

    def test_fallback_places_the_product_without_a_face(self, photo, monkeypatch):
        monkeypatch.setattr(backgrounds, "detect_landmarks", lambda _image: None)
        found = list(backgrounds.image_backgrounds(photo, on_no_face="fallback"))
        assert [b.name for b in found] == ["a.jpg", "b.png"]
        for background in found:
            assert background.landmarks is None
            assert background.roi_rect is not None  # manual ROI mode
            assert background.lash_line.shape[1] == 2

    def test_uses_landmarks_when_a_face_is_detected(self, photo, monkeypatch, synthetic_landmarks):
        scaled = synthetic_landmarks / 400 * np.array([260, 200])
        monkeypatch.setattr(backgrounds, "detect_landmarks", lambda _image: scaled)
        found = list(backgrounds.image_backgrounds(photo, limit=1))
        assert len(found) == 1
        background = found[0]
        assert background.landmarks is not None
        # a detectable face means the pipeline computes its own ROI, so none is forced
        assert background.roi_rect is None
        assert len(background.lash_lines) == 2  # one lash line per eye

    def test_downscales_large_photos(self, tmp_path, monkeypatch):
        monkeypatch.setattr(backgrounds, "detect_landmarks", lambda _image: None)
        directory = tmp_path / "big"
        directory.mkdir()
        cv2.imwrite(str(directory / "wide.png"), np.zeros((400, 2000, 3), np.uint8))
        background = next(iter(backgrounds.image_backgrounds(str(directory), on_no_face="fallback")))
        assert background.image.shape[1] == 900  # keeps ROI under MAX_ROI_WIDTH

    def test_rejects_an_unknown_policy(self, photo):
        with pytest.raises(ValueError):
            list(backgrounds.image_backgrounds(photo, on_no_face="explode"))

    def test_fixed_roi_rule_uses_compute_eye_roi_margins(self):
        """Same margins as the production auto ROI, clipped to the frame."""
        rect = backgrounds.fixed_roi_rect((60, 90, 160, 120), (240, 320))
        assert rect == (
            int(60 - 0.45 * 100),
            int(90 - 2.4 * 30),
            int(160 + 0.45 * 100),
            int(120 + 1.4 * 30),
        )
        assert backgrounds.fixed_roi_rect((10, 10, 110, 40), (240, 320))[:2] == (0, 0)

    def test_roi_rule_is_clipped_to_the_image(self):
        x0, y0, x1, y1 = backgrounds.fixed_roi_rect((0, 0, 320, 240), (240, 320))
        assert (x0, y0) == (0, 0)
        assert (x1, y1) == (320, 240)


class TestImageProduct:
    @pytest.fixture
    def rgba(self) -> np.ndarray:
        rng = np.random.default_rng(3)
        out = np.zeros((40, 80, 4), np.uint8)
        out[10:30, 10:70, :3] = rng.integers(20, 90, (20, 60, 3), dtype=np.uint8)
        out[10:30, 10:70, 3] = 255
        return out

    @pytest.fixture
    def background(self):
        return backgrounds.procedural_backgrounds(1, width=200, height=150, seed=4)[0]

    def test_renders_onto_the_lash_line(self, rgba, background):
        product = ImageProduct(rgba)
        bgr, alpha = product.render(background, Placement())
        assert bgr.shape == (*background.shape, 3)
        assert alpha.shape == background.shape
        assert alpha.max() > 0.9
        ys, xs = np.nonzero(alpha > 0.5)
        line = background.lash_line
        # the product sits on the lash line, not in the middle of the frame
        assert abs(xs.mean() - line[:, 0].mean()) < background.shape[1] * 0.25

    def test_placement_moves_it(self, rgba, background):
        product = ImageProduct(rgba)
        _, base = product.render(background, Placement())
        _, moved = product.render(background, Placement(offset_y=-10))
        assert not np.array_equal(base, moved)
        assert moved.sum() == pytest.approx(base.sum(), rel=0.15)

    def test_declares_its_ground_truth_is_resampled(self, rgba):
        assert ImageProduct(rgba).exact_ground_truth is False
        assert ProceduralProduct().exact_ground_truth is True

    def test_rejects_an_empty_alpha(self, background):
        with pytest.raises(ValueError, match="empty alpha"):
            ImageProduct(np.zeros((10, 10, 4), np.uint8)).render(background, Placement())


class TestLoadProductPng:
    def test_reads_an_rgba_file(self, tmp_path):
        path = tmp_path / "product.png"
        cv2.imwrite(str(path), np.full((8, 8, 4), 200, np.uint8))
        assert load_product_png(str(path)).shape == (8, 8, 4)

    def test_rejects_a_file_without_alpha(self, tmp_path):
        path = tmp_path / "opaque.png"
        cv2.imwrite(str(path), np.zeros((8, 8, 3), np.uint8))
        with pytest.raises(ValueError, match="alpha"):
            load_product_png(str(path))

    def test_rejects_a_missing_file(self, tmp_path):
        with pytest.raises(ValueError, match="could not read"):
            load_product_png(str(tmp_path / "nope.png"))
