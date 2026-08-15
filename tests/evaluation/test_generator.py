"""The generator's whole job is to make ground truth that is true by construction."""

from __future__ import annotations

import json
import os

import numpy as np
import pytest

from evaluation import backgrounds, generator
from evaluation.degrade import Degradation
from evaluation.products import ProceduralProduct
from evaluation.synth import Placement


@pytest.fixture
def background():
    return backgrounds.procedural_backgrounds(1, width=200, height=150, seed=2)[0]


@pytest.fixture
def product():
    return ProceduralProduct(seed=7, n_strands=16)


def _spec(**kwargs) -> generator.CaseSpec:
    defaults = {"case_id": "case_0001", "background": "bg", "product": "p", "condition": "baseline"}
    return generator.CaseSpec(**{**defaults, **kwargs})


class TestBuildCase:
    def test_compositing_equation_holds_exactly(self, background, product):
        case = generator.build_case(background, product, _spec())
        alpha = case.gt_alpha[..., None].astype(np.float64) / 255.0
        expected = case.gt_product[:, :, :3] * alpha + case.bare.astype(np.float64) * (1 - alpha)
        error = np.abs(case.worn.astype(np.float64) - expected)
        # the only discrepancy allowed is 8-bit rounding of alpha, colour and result
        assert error.max() <= 2.0
        assert error.mean() < 0.1

    def test_worn_equals_bare_where_there_is_no_product(self, background, product):
        case = generator.build_case(background, product, _spec())
        empty = case.gt_alpha == 0
        assert empty.any()
        assert np.array_equal(case.worn[empty], case.bare[empty])

    def test_mask_is_the_thresholded_alpha(self, background, product):
        case = generator.build_case(background, product, _spec())
        assert np.array_equal(case.gt_mask > 0, case.gt_alpha >= 128)

    def test_alpha_keeps_semi_transparent_tips(self, background, product):
        case = generator.build_case(background, product, _spec())
        partial = (case.gt_alpha > 5) & (case.gt_alpha < 250)
        assert partial.sum() > 0.15 * (case.gt_alpha > 0).sum()

    def test_ignore_marks_own_lashes_but_never_the_product(self, background, product):
        case = generator.build_case(background, product, _spec())
        assert case.gt_ignore.any()
        assert not (case.gt_ignore & (case.gt_mask > 0)).any()

    def test_shadow_darkens_the_skin_without_entering_ground_truth(self, background, product):
        lit = generator.build_case(background, product, _spec())
        shadowed = generator.build_case(background, product, _spec(shadow=0.5))
        assert np.array_equal(lit.gt_alpha, shadowed.gt_alpha)
        outside = lit.gt_alpha == 0
        assert shadowed.worn[outside].mean() < lit.worn[outside].mean()

    def test_placement_moves_the_product(self, background, product):
        base = generator.build_case(background, product, _spec())
        moved = generator.build_case(background, product, _spec(placement=Placement(offset_y=-6)))
        assert not np.array_equal(base.gt_alpha, moved.gt_alpha)
        assert moved.gt_alpha.sum() == pytest.approx(base.gt_alpha.sum(), rel=0.1)

    def test_degradation_hits_the_worn_image_only(self, background, product):
        case = generator.build_case(background, product, _spec(worn=Degradation(brightness=0.6)))
        assert case.worn.mean() < case.bare.mean() * 0.8

    def test_bare_and_worn_noise_are_independent(self, background, product):
        noisy = Degradation(noise_sigma=6.0)
        case = generator.build_case(background, product, _spec(worn=noisy, bare=noisy))
        clean = generator.build_case(background, product, _spec())
        empty = clean.gt_alpha == 0
        # if both images shared one noise field, the difference would cancel here
        difference = np.abs(case.worn[empty].astype(int) - case.bare[empty].astype(int)).mean()
        assert difference > 4.0

    def test_misalignment_moves_the_bare_image(self, background, product):
        aligned = generator.build_case(background, product, _spec())
        shifted = generator.build_case(background, product, _spec(bare_misalign_px=5.0))
        assert np.abs(shifted.bare.astype(int) - aligned.bare.astype(int)).mean() > 1.0
        assert np.array_equal(shifted.gt_alpha, aligned.gt_alpha)

    def test_is_reproducible(self, background, product):
        a = generator.build_case(background, product, _spec(worn=Degradation(noise_sigma=5.0)))
        b = generator.build_case(background, product, _spec(worn=Degradation(noise_sigma=5.0)))
        assert np.array_equal(a.worn, b.worn)


class TestWriteCase:
    def test_writes_images_and_metadata(self, tmp_path, background, product):
        case = generator.build_case(background, product, _spec(placement=Placement(rotation_deg=5)))
        directory = generator.write_case(str(tmp_path), case)
        expected = ("bare.png", "worn.png", "gt_alpha.png", "gt_mask.png", "gt_ignore.png", "gt_product.png")
        for name in expected:
            assert os.path.exists(os.path.join(directory, name)), name
        with open(os.path.join(directory, "metadata.json"), encoding="utf-8") as handle:
            meta = json.load(handle)
        assert meta["id"] == "case_0001"
        assert meta["rotation_deg"] == 5
        assert meta["condition"] == "baseline"
        assert meta["roi_rect"] is not None  # procedural eyes have no detectable face

    def test_saved_alpha_survives_the_round_trip(self, tmp_path, background, product):
        import cv2

        case = generator.build_case(background, product, _spec())
        directory = generator.write_case(str(tmp_path), case)
        loaded = cv2.imread(os.path.join(directory, "gt_alpha.png"), cv2.IMREAD_GRAYSCALE)
        assert np.array_equal(loaded, case.gt_alpha)


class TestPlanCases:
    def test_covers_one_axis_at_a_time(self):
        specs = generator.plan_cases(["bg_a", "bg_b"], ["prod"], count=12)
        assert len(specs) == 12
        assert {s.case_id for s in specs} == {f"case_{i + 1:04d}" for i in range(12)}
        assert specs[0].condition == "baseline"
        assert len({s.condition for s in specs}) > 3

    def test_every_condition_shares_a_background_with_its_baseline(self):
        """Otherwise a condition is compared against a *different* eye, and the
        condition breakdown measures background difficulty instead of the condition."""
        specs = generator.plan_cases([f"bg_{i}" for i in range(3)], ["p"], count=generator.BLOCK_SIZE * 3)
        baselines = {s.pair_key for s in specs if s.condition == "baseline"}
        for spec in specs:
            assert spec.pair_key in baselines, f"{spec.case_id} has no baseline for {spec.pair_key}"

    def test_a_block_holds_every_variant_exactly_once(self):
        specs = generator.plan_cases(["bg"], ["p"], count=generator.BLOCK_SIZE)
        conditions = [(s.condition, str(s.condition_value)) for s in specs]
        assert len(conditions) == len(set(conditions))
        assert sum(1 for s in specs if s.condition == "baseline") == 1

    def test_is_deterministic(self):
        a = generator.plan_cases(["bg"], ["p"], count=9, seed=4)
        b = generator.plan_cases(["bg"], ["p"], count=9, seed=4)
        assert [s.as_dict() for s in a] == [s.as_dict() for s in b]

    def test_seed_changes_which_pairs_come_first(self):
        names = [f"bg_{i}" for i in range(6)]
        first = generator.plan_cases(names, ["p"], count=6, seed=1)[0].background
        others = {generator.plan_cases(names, ["p"], count=6, seed=s)[0].background for s in range(8)}
        assert len(others) > 1, "the seed argument must actually do something"
        assert first == generator.plan_cases(names, ["p"], count=6, seed=1)[0].background

    def test_uses_every_background_when_the_budget_allows(self):
        specs = generator.plan_cases([f"bg_{i}" for i in range(5)], ["p"], count=generator.BLOCK_SIZE * 5)
        assert len({s.background for s in specs}) == 5

    def test_reports_how_many_blocks_are_complete(self):
        assert generator.complete_blocks(generator.plan_cases(["a", "b"], ["p"], count=generator.BLOCK_SIZE))
        partial = generator.plan_cases(["a", "b"], ["p"], count=generator.BLOCK_SIZE + 3)
        assert generator.complete_blocks(partial) == 1

    def test_rejects_empty_inputs(self):
        with pytest.raises(ValueError):
            generator.plan_cases([], ["p"], count=3)
