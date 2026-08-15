"""Aggregation and reporting: summary.json, summary.csv and report.md.

Averages ignore NaN (an undefined ratio contributes nothing instead of a fake 0 or 1)
and every table states how many runs it is based on. No pass/fail thresholds are applied:
the point of this benchmark is to establish a baseline, not to award a grade.
"""

from __future__ import annotations

import csv
import json
import os
from collections.abc import Iterable, Sequence
from datetime import UTC, datetime
from typing import Any

import numpy as np

HEADLINE = (
    "dice",
    "iou",
    "precision",
    "recall",
    "dice_ex_own",
    "precision_ex_own",
    "mad",
    "grad",
    "boundary_f1",
    "rgb_mae",
    "rgb_rmse",
    "recompose_mae",
    "reconstruction_error",
    "seconds",
)


def _mean(rows: Sequence[dict[str, Any]], key: str) -> float:
    values = [row[key] for row in rows if isinstance(row.get(key), int | float)]
    finite = [float(v) for v in values if not np.isnan(float(v))]
    return float(np.mean(finite)) if finite else float("nan")


def _summarise(rows: Sequence[dict[str, Any]], keys: Iterable[str] = HEADLINE) -> dict[str, float]:
    summary = {key: _mean(rows, key) for key in keys}
    summary["runs"] = float(len(rows))
    summary["failed"] = float(sum(1 for row in rows if row.get("failed")))
    return summary


def _group(rows: Sequence[dict[str, Any]], key: str) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row.get(key)), []).append(row)
    return dict(sorted(grouped.items()))


def summarise(
    rows: Sequence[dict[str, Any]],
    mutation_rows: Sequence[dict[str, Any]] = (),
    roi_rows: Sequence[dict[str, Any]] = (),
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    by_run: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_run.setdefault(f"{row['mode']}/{row['brush']}", []).append(row)
    by_run = dict(sorted(by_run.items()))
    auto = [row for row in rows if row["brush"] == "auto"]
    return {
        "generated_at": datetime.now(UTC).isoformat(),
        "cases": len({row["case_id"] for row in rows}),
        "runs": len(rows),
        "config": config or {},
        "overall": {name: _summarise(group) for name, group in by_run.items() if group},
        "by_condition": {
            name: {
                run_name: _summarise(
                    [r for r in group if f"{r['mode']}/{r['brush']}" == run_name],
                    ("dice", "iou", "precision", "recall", "mad", "grad", "rgb_mae"),
                )
                for run_name in sorted({f"{r['mode']}/{r['brush']}" for r in group})
            }
            for name, group in _group(rows, "condition").items()
        },
        "by_mode_condition": {
            name: _summarise(group, ("dice", "iou", "precision", "recall", "mad", "grad"))
            for name, group in _group(auto, "condition").items()
        },
        "pixel_mutation": list(mutation_rows),
        "roi_downscale": list(roi_rows),
        "best_cases": _ranked(auto, "dice", best=True),
        "worst_cases": _ranked(auto, "dice", best=False),
    }


def _rounded(row: dict[str, Any], key: str) -> float | None:
    value = row.get(key)
    if not isinstance(value, int | float) or np.isnan(float(value)):
        return None
    return round(float(value), 4)


def _ranked(rows: Sequence[dict[str, Any]], key: str, best: bool, limit: int = 5) -> list[dict[str, Any]]:
    usable = [row for row in rows if _rounded(row, key) is not None]
    ordered = sorted(usable, key=lambda row: float(row[key]), reverse=best)
    return [
        {
            "case_id": row["case_id"],
            "mode": row["mode"],
            "condition": row.get("condition"),
            "condition_value": row.get("condition_value"),
            key: _rounded(row, key),
            "recall": _rounded(row, "recall"),
            "precision": _rounded(row, "precision"),
        }
        for row in ordered[:limit]
    ]


def write_csv(path: str, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _table(header: Sequence[str], rows: Iterable[Sequence[Any]]) -> str:
    lines = [f"| {' | '.join(header)} |", f"| {' | '.join('---' for _ in header)} |"]
    for row in rows:
        lines.append(f"| {' | '.join(_fmt(value) for value in row)} |")
    return "\n".join(lines)


def _fmt(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        if np.isnan(value):
            return "n/a"
        return f"{value:.4f}" if abs(value) < 1000 else f"{value:.1f}"
    return str(value)


def render_markdown(summary: dict[str, Any]) -> str:
    parts: list[str] = [
        "# Synthetic Benchmark — baseline",
        "",
        f"- generated: `{summary['generated_at']}`",
        f"- cases: **{summary['cases']}**, pipeline runs: **{summary['runs']}**",
        "- ground truth is synthetic. These numbers do **not** prove real-world quality;",
        "  see `evaluation/README.md` for what synthetic data cannot reproduce.",
        "- binarisation threshold: `alpha >= 0.5`. `*_ex_own` excludes the model's own",
        "  lashes, which the pipeline is meant to pick up but which are not the product.",
        "",
        "### How to quote these numbers (`evaluation/README.md` §0)",
        "",
        "| table | layer | how far it carries |",
        "| --- | --- | --- |",
        "| Overall / By condition / Best-worst | **C** absolute score, **B** relative trend |"
        " absolute values are a regression baseline only — never quote them as the system's"
        " accuracy. The *ranking* of conditions is trustworthy |",
        "| Pixel mutation / ROI downscale | **A** property of the production code path |"
        " holds for real photos too; actionable as-is |",
        "",
        "Ground truth here is per-strand alpha covering only a few percent of the frame, so a"
        " one-pixel spill halves precision. Read `reconstruction_error` and"
        " `cases/*/comparison_*.png` next to the scores: the gap between them is the gap"
        " between per-strand agreement and what a human sees.",
        "",
        "## Overall",
        "",
        _table(
            [
                "mode / brush",
                "runs",
                "failed",
                "Dice",
                "IoU",
                "Precision",
                "Recall",
                "Dice(ex own)",
                "MAD",
                "Grad",
            ],
            [
                [
                    name,
                    int(values["runs"]),
                    int(values["failed"]),
                    values["dice"],
                    values["iou"],
                    values["precision"],
                    values["recall"],
                    values["dice_ex_own"],
                    values["mad"],
                    values["grad"],
                ]
                for name, values in summary["overall"].items()
            ],
        ),
        "",
        "## Product fidelity and recomposition",
        "",
        _table(
            ["mode / brush", "RGB MAE", "RGB RMSE", "recompose MAE", "reconstruction err", "sec/run"],
            [
                [
                    name,
                    values["rgb_mae"],
                    values["rgb_rmse"],
                    values["recompose_mae"],
                    values["reconstruction_error"],
                    values["seconds"],
                ]
                for name, values in summary["overall"].items()
            ],
        ),
        "",
        "## By condition (one axis changed at a time)",
        "",
    ]
    conditions = summary["by_condition"]
    run_names = sorted({name for group in conditions.values() for name in group})
    parts.append(
        _table(
            ["condition", *[f"Dice {name}" for name in run_names], "Grad (bare/auto)"],
            [
                [
                    name,
                    *[group.get(run, {}).get("dice", float("nan")) for run in run_names],
                    group.get("bare/auto", {}).get("grad", float("nan")),
                ]
                for name, group in conditions.items()
            ],
        )
    )
    if summary.get("pixel_mutation"):
        parts += [
            "",
            "## Pixel mutation (product RGB under a transform)",
            "",
            "`production_default` marks the interpolation the shipped recompose path uses.",
            "",
            _table(
                [
                    "transform",
                    "variant",
                    "exact preserved",
                    "mutation rate",
                    "RGB MAE",
                    "fringe RGB MAE",
                    "alpha MAD",
                    "alpha Grad",
                ],
                [
                    [
                        row["transform"],
                        row["variant"] + (" *" if row.get("production_default") else ""),
                        row["exact_color_preservation_rate"],
                        row["rgb_mutation_rate"],
                        row["rgb_mae"],
                        row.get("fringe_rgb_mae"),
                        row["alpha_mad"],
                        row["alpha_grad"],
                    ]
                    for row in summary["pixel_mutation"]
                ],
            ),
        ]
    if summary.get("roi_downscale"):
        parts += [
            "",
            "## ROI downscale (crop_roi, before extraction starts)",
            "",
            _table(
                ["target ROI width", "scale", "interpolation", "exact preserved", "RGB MAE"],
                [
                    [
                        row["roi_width"],
                        row["scale"],
                        row["interpolation"],
                        row["exact_color_preservation_rate"],
                        row["rgb_mae"],
                    ]
                    for row in summary["roi_downscale"]
                ],
            ),
        ]
    for label, key in (("Best cases", "best_cases"), ("Worst cases", "worst_cases")):
        parts += [
            "",
            f"## {label} (bare & worn-only, no brush)",
            "",
            _table(
                ["case", "mode", "condition", "value", "Dice", "Precision", "Recall"],
                [
                    [
                        row["case_id"],
                        row["mode"],
                        row["condition"],
                        row["condition_value"],
                        row["dice"],
                        row["precision"],
                        row["recall"],
                    ]
                    for row in summary[key]
                ],
            ),
        ]
    return "\n".join(parts) + "\n"


def write_report(output_root: str, rows: Sequence[dict[str, Any]], summary: dict[str, Any]) -> None:
    os.makedirs(output_root, exist_ok=True)
    with open(os.path.join(output_root, "summary.json"), "w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2, default=str)
    write_csv(os.path.join(output_root, "summary.csv"), rows)
    with open(os.path.join(output_root, "report.md"), "w", encoding="utf-8") as handle:
        handle.write(render_markdown(summary))
