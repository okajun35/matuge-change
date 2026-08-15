"""End-to-end: generate a tiny dataset, run the real pipeline over it, write a report.

Kept deliberately small (2 backgrounds, 1 product, a handful of cases, ~100 px images)
so CI needs no external dataset and closed-form matting stays fast.
"""

from __future__ import annotations

import json
import os

import numpy as np
import pytest

from evaluation import backgrounds, report
from evaluation.dataset import load_dataset
from evaluation.generator import build_case, plan_cases, write_case, write_manifest
from evaluation.pipeline import oracle_constraints, run_pipeline, to_roi_space
from evaluation.products import ProceduralProduct
from evaluation.runner import RunConfig, evaluate_case, run_dataset


@pytest.fixture(scope="module")
def dataset(tmp_path_factory) -> str:
    root = str(tmp_path_factory.mktemp("dataset"))
    sources = backgrounds.procedural_backgrounds(2, width=120, height=96, seed=5)
    product = ProceduralProduct(seed=2, n_strands=14)
    by_name = {background.name: background for background in sources}
    specs = plan_cases(list(by_name), [product.name], count=5, seed=0)
    for spec in specs:
        write_case(root, build_case(by_name[spec.background], product, spec))
    write_manifest(root, specs)
    return root


class TestManifest:
    def test_only_the_listed_cases_are_loaded(self, tmp_path):
        """Regenerating a smaller dataset must not inherit the previous, larger one."""
        root = str(tmp_path)
        sources = backgrounds.procedural_backgrounds(1, width=96, height=72, seed=1)
        product = ProceduralProduct(seed=1, n_strands=8)
        by_name = {b.name: b for b in sources}
        many = plan_cases(list(by_name), [product.name], count=4, seed=0)
        for spec in many:
            write_case(root, build_case(by_name[spec.background], product, spec))
        write_manifest(root, many)
        assert len(load_dataset(root)) == 4

        few = many[:2]
        write_manifest(root, few)  # the other two folders are still on disk
        assert [case.case_id for case in load_dataset(root)] == [s.case_id for s in few]

    def test_a_missing_case_is_an_error_not_a_silent_gap(self, tmp_path):
        root = str(tmp_path)
        sources = backgrounds.procedural_backgrounds(1, width=96, height=72, seed=1)
        product = ProceduralProduct(seed=1, n_strands=8)
        by_name = {b.name: b for b in sources}
        specs = plan_cases(list(by_name), [product.name], count=2, seed=0)
        write_case(root, build_case(by_name[specs[0].background], product, specs[0]))
        write_manifest(root, specs)
        with pytest.raises(FileNotFoundError, match="missing"):
            load_dataset(root)


class TestDatasetRoundTrip:
    def test_loads_every_generated_case(self, dataset):
        cases = load_dataset(dataset)
        assert len(cases) == 5
        assert [case.case_id for case in cases] == [f"case_{i + 1:04d}" for i in range(5)]

    def test_a_case_exposes_images_and_a_manual_roi(self, dataset):
        case = load_dataset(dataset)[0]
        assert case.worn.shape == (96, 120, 3)
        assert case.bare is not None
        assert case.gt_alpha.shape == (96, 120)
        assert case.gt_ignore is not None
        assert case.roi_rect is not None  # procedural eyes have no detectable face

    def test_missing_dataset_is_reported(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_dataset(str(tmp_path / "nope"))


class TestPipelineAdapter:
    def test_extracts_something_in_both_modes(self, dataset):
        case = load_dataset(dataset)[0]
        with_bare = run_pipeline(case.worn, case.bare, case.roi_rect)
        worn_only = run_pipeline(case.worn, None, case.roi_rect)
        assert with_bare.mode == "bare" and worn_only.mode == "worn_only"
        assert with_bare.roi_b is not None and worn_only.roi_b is None
        for prediction in (with_bare, worn_only):
            assert prediction.alpha.shape == prediction.roi_a.shape[:2]
            assert prediction.product_rgba.shape[2] == 4
            assert 0.0 <= prediction.alpha.min() <= prediction.alpha.max() <= 1.0

    def test_oracle_constraints_pin_the_trimap(self, dataset):
        case = load_dataset(dataset)[0]
        prediction = run_pipeline(
            case.worn,
            case.bare,
            case.roi_rect,
            lambda roi: oracle_constraints(to_roi_space(case.gt_alpha, roi)),
        )
        gt_roi = to_roi_space(case.gt_alpha, prediction.roi)
        # deep inside the product the alpha must be solid, because the brush said so
        deep = gt_roi == 255
        if deep.any():
            assert prediction.alpha[deep].mean() > 0.5

    def test_oracle_brush_raises_overlap_and_precision(self, dataset):
        """The automatic estimate is recall-heavy; a perfect brush trades that for Dice.

        Recall may well *drop*, because the automatic trimap floods the unknown band and
        catches almost everything at the cost of precision.
        """
        case = load_dataset(dataset)[0]
        rows = evaluate_case(case, RunConfig(modes=("bare",), brushes=("auto", "oracle"), save_images=False))
        by_brush = {row["brush"]: row for row in rows}
        assert by_brush["oracle"]["dice"] > by_brush["auto"]["dice"]
        assert by_brush["oracle"]["precision"] > by_brush["auto"]["precision"]
        assert by_brush["oracle"]["mad"] < by_brush["auto"]["mad"]

    def test_oracle_ignores_the_probability_map_by_design(self, dataset):
        """With ground-truth strokes the trimap is fully pinned, so both modes agree.

        That makes the oracle rows a measurement of the matting step alone.
        """
        case = load_dataset(dataset)[0]
        rows = evaluate_case(
            case, RunConfig(modes=("bare", "worn_only"), brushes=("oracle",), save_images=False)
        )
        assert rows[0]["dice"] == pytest.approx(rows[1]["dice"], abs=1e-6)


class TestRunnerAndReport:
    def test_full_run_writes_every_artefact(self, dataset, tmp_path):
        cases = load_dataset(dataset)[:2]
        config = RunConfig(modes=("bare", "worn_only"), brushes=("auto",))
        output = str(tmp_path / "results")
        rows = run_dataset(cases, config, output)
        assert len(rows) == 4  # 2 cases x 2 modes
        for row in rows:
            assert 0.0 <= row["dice"] <= 1.0
            assert row["mad"] >= 0.0
            assert row["condition"]

        summary = report.summarise(rows, config={"dataset": dataset})
        report.write_report(output, rows, summary)
        for name in ("summary.json", "summary.csv", "report.md"):
            assert os.path.exists(os.path.join(output, name)), name
        case_dir = os.path.join(output, "cases", cases[0].case_id)
        for name in (
            "predicted_alpha_bare_auto.png",
            "predicted_product_bare_auto.png",
            "gt_mask.png",
            "comparison_bare_auto.png",
        ):
            assert os.path.exists(os.path.join(case_dir, name)), name

    def test_report_contains_the_headline_numbers(self, dataset, tmp_path):
        rows = run_dataset(
            load_dataset(dataset)[:1],
            RunConfig(modes=("worn_only",), brushes=("auto",), save_images=False),
            None,
        )
        summary = report.summarise(rows)
        text = report.render_markdown(summary)
        assert "Dice" in text and "IoU" in text and "Precision" in text and "Recall" in text
        assert "worn_only/auto" in text
        assert "do **not** prove real-world quality" in text
        # a reader who only ever sees report.md must still learn which numbers carry over
        assert "How to quote these numbers" in text
        assert "regression baseline only" in text
        assert "property of the production code path" in text

        output = str(tmp_path / "out")
        report.write_report(output, rows, summary)
        with open(os.path.join(output, "summary.json"), encoding="utf-8") as handle:
            saved = json.load(handle)
        assert saved["cases"] == 1
        assert "worn_only/auto" in saved["overall"]

    def test_comparison_image_has_seven_panels(self, dataset):
        from evaluation.runner import comparison_image

        case = load_dataset(dataset)[0]
        prediction = run_pipeline(case.worn, case.bare, case.roi_rect)
        from evaluation.pipeline import composite_on, to_image_space

        shape = case.gt_alpha.shape
        alpha_full = to_image_space(prediction.alpha, prediction.roi, shape)
        product_full = to_image_space(prediction.product_rgba, prediction.roi, shape)
        recomposed = to_image_space(
            composite_on(prediction, to_roi_space(case.bare, prediction.roi)), prediction.roi, shape
        )
        image = comparison_image(case, alpha_full, product_full, recomposed)
        assert image.shape == (shape[0], shape[1] * 7, 3)


class TestFailuresAreRecorded:
    def test_a_degenerate_case_completes_with_an_empty_matte(self, tmp_path):
        """A flat image gives a probability map that never reaches fg_thresh. Matting used to
        raise inside pymatting on such a trimap; it now short-circuits to an empty matte."""
        import cv2

        directory = tmp_path / "case_0001"
        directory.mkdir()
        flat = np.full((80, 100, 3), 128, np.uint8)
        cv2.imwrite(str(directory / "worn.png"), flat)
        cv2.imwrite(str(directory / "bare.png"), flat)
        cv2.imwrite(str(directory / "gt_alpha.png"), np.zeros((80, 100), np.uint8))
        cv2.imwrite(str(directory / "gt_mask.png"), np.zeros((80, 100), np.uint8))
        (directory / "metadata.json").write_text(
            json.dumps({"id": "case_0001", "condition": "degenerate", "roi_rect": [10, 10, 90, 70]})
        )

        rows = run_dataset(
            load_dataset(str(tmp_path)),
            RunConfig(modes=("bare",), brushes=("auto",), save_images=False),
            None,
        )
        assert len(rows) == 1
        assert rows[0]["failed"] is False
        # nothing is extracted, which is the right answer for a flat image
        assert rows[0]["reconstruction_error"] == 0.0

    def test_a_pipeline_exception_becomes_a_row(self, tmp_path, monkeypatch):
        """A crash in one case is a result, not a reason to abandon the whole run."""
        import cv2

        from evaluation import runner as evaluation_runner

        directory = tmp_path / "case_0001"
        directory.mkdir()
        flat = np.full((80, 100, 3), 128, np.uint8)
        cv2.imwrite(str(directory / "worn.png"), flat)
        cv2.imwrite(str(directory / "bare.png"), flat)
        cv2.imwrite(str(directory / "gt_alpha.png"), np.zeros((80, 100), np.uint8))
        cv2.imwrite(str(directory / "gt_mask.png"), np.zeros((80, 100), np.uint8))
        (directory / "metadata.json").write_text(
            json.dumps({"id": "case_0001", "condition": "degenerate", "roi_rect": [10, 10, 90, 70]})
        )

        def explode(*_args, **_kwargs):
            raise ValueError("matting exploded")

        monkeypatch.setattr(evaluation_runner, "run_pipeline", explode)

        rows = run_dataset(
            load_dataset(str(tmp_path)),
            RunConfig(modes=("bare",), brushes=("auto",), save_images=False),
            None,
        )
        assert len(rows) == 1
        assert rows[0]["failed"] is True
        assert "matting exploded" in rows[0]["error"]
        summary = report.summarise(rows)
        assert summary["overall"]["bare/auto"]["failed"] == 1
        assert np.isnan(summary["overall"]["bare/auto"]["dice"])


class TestRecompositionReference:
    def test_reference_is_the_ground_truth_product_on_the_same_base(self, dataset):
        """Comparing against worn.png would fold that photo's JPEG/noise/shadow into the
        score — on a JPEG case it made the metric read below the floor those artefacts
        alone set, i.e. lower than any error it could detect."""
        from evaluation.runner import ideal_recomposition

        case = load_dataset(dataset)[0]
        reference = ideal_recomposition(case)
        alpha = case.gt_alpha[..., None].astype(np.float64) / 255.0
        expected = alpha * case.gt_product[:, :, :3] + (1 - alpha) * case.bare
        assert np.abs(reference.astype(np.float64) - expected).max() <= 1.0

    def test_perfect_extraction_would_score_zero(self, dataset):
        from evaluation import metrics
        from evaluation.runner import ideal_recomposition

        case = load_dataset(dataset)[0]
        reference = ideal_recomposition(case)
        result = metrics.region_fidelity(reference, reference, case.gt_mask > 0)
        assert result["mae"] == pytest.approx(0.0)

    def test_falls_back_to_worn_without_a_ground_truth_product(self, tmp_path):
        import cv2

        from evaluation.runner import ideal_recomposition

        directory = tmp_path / "case_0001"
        directory.mkdir()
        worn = np.full((16, 16, 3), 90, np.uint8)
        cv2.imwrite(str(directory / "worn.png"), worn)
        cv2.imwrite(str(directory / "gt_alpha.png"), np.zeros((16, 16), np.uint8))
        (directory / "metadata.json").write_text(json.dumps({"id": "case_0001"}))
        case = load_dataset(str(tmp_path))[0]
        assert np.array_equal(ideal_recomposition(case), worn)


class TestAggregation:
    def test_condition_values_are_not_averaged_together(self):
        rows = [
            {
                "case_id": f"c{i}",
                "mode": "bare",
                "brush": "auto",
                "condition": "rotation_deg",
                "condition_value": value,
                "pair_key": "bg|p",
                "dice": dice,
            }
            for i, (value, dice) in enumerate([(5.0, 0.8), (10.0, 0.2)])
        ]
        summary = report.summarise(rows)
        assert set(summary["by_condition"]) == {"rotation_deg = 5.0", "rotation_deg = 10.0"}
        assert summary["by_condition"]["rotation_deg = 10.0"]["bare/auto"]["dice"] == pytest.approx(0.2)

    def test_paired_delta_removes_the_background_effect(self):
        """Two eyes of different difficulty, each with its own baseline and blur case."""
        rows = [
            {"case_id": "a1", "pair_key": "easy|p", "condition": "baseline", "dice": 0.9},
            {"case_id": "a2", "pair_key": "easy|p", "condition": "blur_sigma", "dice": 0.8},
            {"case_id": "b1", "pair_key": "hard|p", "condition": "baseline", "dice": 0.5},
            {"case_id": "b2", "pair_key": "hard|p", "condition": "blur_sigma", "dice": 0.4},
        ]
        for row in rows:
            row.update({"mode": "bare", "brush": "auto", "condition_value": None})
        summary = report.summarise(rows)
        # raw means would say blur scores 0.60 against a 0.70 baseline; paired says -0.10
        assert summary["paired_delta"]["blur_sigma = None"]["dice_delta"] == pytest.approx(-0.1)
        assert summary["paired_delta"]["blur_sigma = None"]["pairs"] == 2

    def test_success_rate_is_reported_next_to_the_averages(self):
        rows = [
            {"case_id": "a", "mode": "bare", "brush": "auto", "condition": "baseline", "dice": 1.0},
            {
                "case_id": "b",
                "mode": "bare",
                "brush": "auto",
                "condition": "baseline",
                "dice": float("nan"),
                "failed": True,
            },
        ]
        summary = report.summarise(rows)
        # the mean skips the crash, so the success rate is what stops that being a lie
        assert summary["overall"]["bare/auto"]["dice"] == pytest.approx(1.0)
        assert summary["overall"]["bare/auto"]["success_rate"] == pytest.approx(0.5)
        assert "success" in report.render_markdown(summary)

    def test_summary_json_is_parseable_by_a_strict_reader(self, tmp_path):
        rows = [
            {
                "case_id": "a",
                "mode": "bare",
                "brush": "auto",
                "condition": "baseline",
                "dice": float("nan"),
            }
        ]
        output = str(tmp_path / "out")
        report.write_report(output, rows, report.summarise(rows))
        with open(os.path.join(output, "summary.json"), encoding="utf-8") as handle:
            text = handle.read()
        assert "NaN" not in text  # json.load accepts NaN, browsers and jq do not
        assert json.loads(text)["overall"]["bare/auto"]["dice"] is None

    def test_nan_metrics_do_not_pollute_the_mean(self):
        rows = [
            {"case_id": "a", "mode": "bare", "brush": "auto", "condition": "baseline", "dice": 0.8},
            {
                "case_id": "b",
                "mode": "bare",
                "brush": "auto",
                "condition": "baseline",
                "dice": float("nan"),
            },
        ]
        summary = report.summarise(rows)
        assert summary["overall"]["bare/auto"]["dice"] == pytest.approx(0.8)

    def test_condition_breakdown_splits_by_axis(self):
        rows = [
            {"case_id": "a", "mode": "bare", "brush": "auto", "condition": "baseline", "dice": 0.9},
            {"case_id": "b", "mode": "bare", "brush": "auto", "condition": "rotation_deg", "dice": 0.5},
        ]
        summary = report.summarise(rows)
        assert set(summary["by_condition"]) == {"baseline", "rotation_deg"}
        assert summary["by_condition"]["rotation_deg"]["bare/auto"]["dice"] == pytest.approx(0.5)
        assert np.isnan(summary["by_condition"]["baseline"]["bare/auto"]["rgb_mae"])
