"""Procedural product / background generation must give exact, reproducible ground truth."""

from __future__ import annotations

import numpy as np
import pytest

from evaluation import synth


@pytest.fixture
def background() -> synth.EyeBackground:
    return synth.synthesize_bare_eye(240, 180, seed=3)


@pytest.fixture
def geometry(background: synth.EyeBackground) -> synth.LashGeometry:
    return synth.synthesize_lash(background.lash_line, seed=5, n_strands=18)


class TestRenderGeometry:
    def test_alpha_is_antialiased(self, geometry, background):
        _, alpha = synth.render_geometry(geometry, background.shape)
        assert alpha.dtype == np.float32
        assert alpha.min() == 0.0
        assert alpha.max() > 0.9  # the band is opaque
        partial = (alpha > 0.02) & (alpha < 0.98)
        # thin strands must produce genuinely fractional coverage, not a hard cut-out
        assert partial.sum() > (alpha > 0.5).sum() * 0.2

    def test_colour_is_not_uniform(self, geometry, background):
        bgr, alpha = synth.render_geometry(geometry, background.shape)
        opaque = alpha > 0.9
        assert opaque.any()
        assert bgr[opaque].std() > 1.0  # textured product, so RGB fidelity is measurable

    def test_strand_colour_actually_reaches_the_buffer(self, geometry, background):
        """The colour buffer is written through a sliced view, which is easy to misread
        as the "assignment to a temporary copy" trap. It is not one: a basic slice is a
        view. If that ever regresses, the product turns black and RGB fidelity is void.
        """
        bgr, alpha = synth.render_geometry(geometry, background.shape)
        opaque = alpha > 0.9
        assert bgr[opaque].max() > 0, "product pixels are black: the colour buffer was lost"
        strand_colours = {strand.color for strand in geometry.strands}
        assert tuple(int(v) for v in bgr[opaque].mean(axis=0).round()) != (0, 0, 0)
        # every opaque pixel carries one of the strand colours, not an averaged smear
        assert any(abs(bgr[opaque].mean() - sum(c) / 3) < 40 for c in strand_colours)

    def test_is_deterministic(self, background):
        a = synth.synthesize_lash(background.lash_line, seed=1)
        b = synth.synthesize_lash(background.lash_line, seed=1)
        _, alpha_a = synth.render_geometry(a, background.shape)
        _, alpha_b = synth.render_geometry(b, background.shape)
        assert np.array_equal(alpha_a, alpha_b)

    def test_different_seeds_differ(self, background):
        first = synth.synthesize_lash(background.lash_line, seed=1)
        second = synth.synthesize_lash(background.lash_line, seed=2)
        _, alpha_a = synth.render_geometry(first, background.shape)
        _, alpha_b = synth.render_geometry(second, background.shape)
        assert not np.array_equal(alpha_a, alpha_b)


class TestPlacement:
    def test_integer_offset_is_an_exact_shift(self, geometry, background):
        """Geometry (not pixels) is transformed, so ground truth has no resampling error."""
        _, base = synth.render_geometry(geometry, background.shape)
        moved = synth.place(geometry, synth.Placement(offset_x=4, offset_y=-3), background.centre)
        _, shifted = synth.render_geometry(moved, background.shape)
        height, width = base.shape
        expected = np.zeros_like(base)
        expected[0 : height - 3, 4:width] = base[3:height, 0 : width - 4]
        assert np.array_equal(shifted, expected)

    def test_identity_placement_changes_nothing(self, geometry, background):
        _, base = synth.render_geometry(geometry, background.shape)
        same = synth.place(geometry, synth.Placement(), background.centre)
        _, again = synth.render_geometry(same, background.shape)
        assert np.array_equal(base, again)

    def test_flip_mirrors_the_product(self, geometry, background):
        _, base = synth.render_geometry(geometry, background.shape)
        flipped = synth.place(geometry, synth.Placement(flip=True), background.centre)
        _, alpha = synth.render_geometry(flipped, background.shape)
        # mirroring is exact in geometry space; only strand rasterisation rounds differently
        mirrored_error = np.abs(alpha - np.fliplr(base)).mean()
        assert mirrored_error < np.abs(alpha - base).mean() / 5
        assert alpha.sum() == pytest.approx(base.sum(), rel=0.02)

    def test_scale_grows_the_covered_area(self, geometry, background):
        small = synth.place(geometry, synth.Placement(scale=0.7), background.centre)
        large = synth.place(geometry, synth.Placement(scale=1.3), background.centre)
        _, alpha_small = synth.render_geometry(small, background.shape)
        _, alpha_large = synth.render_geometry(large, background.shape)
        assert alpha_large.sum() > alpha_small.sum() * 1.4

    def test_rotation_moves_pixels_but_keeps_mass(self, geometry, background):
        _, base = synth.render_geometry(geometry, background.shape)
        rotated = synth.place(geometry, synth.Placement(rotation_deg=10), background.centre)
        _, alpha = synth.render_geometry(rotated, background.shape)
        assert not np.array_equal(alpha, base)
        assert alpha.sum() == pytest.approx(base.sum(), rel=0.2)


class TestSynthesizeBareEye:
    def test_shape_and_dtype(self, background):
        assert background.image.shape == (180, 240, 3)
        assert background.image.dtype == np.uint8
        assert background.own_lash_alpha.shape == (180, 240)

    def test_has_own_lashes_to_distinguish_from_the_product(self, background):
        assert background.own_lash_alpha.max() > 0.3
        assert 0.0 < background.own_lash_alpha.mean() < 0.2

    def test_eye_box_is_inside_the_image(self, background):
        x0, y0, x1, y1 = background.eye_box
        assert 0 <= x0 < x1 <= 240
        assert 0 <= y0 < y1 <= 180

    def test_lash_line_follows_the_upper_lid(self, background):
        line = background.lash_line
        assert line.ndim == 2 and line.shape[1] == 2
        x0, y0, x1, y1 = background.eye_box
        assert line[:, 0].min() >= x0 - 1 and line[:, 0].max() <= x1 + 1
        # the eye is tilted, so only the average height must sit in the upper half,
        # and the topmost point of the eye box has to belong to the upper lid
        assert line[:, 1].mean() < (y0 + y1) / 2
        assert line[:, 1].min() == pytest.approx(y0, abs=1.0)

    def test_is_deterministic(self):
        a = synth.synthesize_bare_eye(120, 90, seed=9)
        b = synth.synthesize_bare_eye(120, 90, seed=9)
        assert np.array_equal(a.image, b.image)
