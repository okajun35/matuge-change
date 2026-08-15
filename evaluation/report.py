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
    "grad_mean",
    "boundary_f1",
    "rgb_mae",
    "rgb_rmse",
    "opaque_coverage",
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
    failed = sum(1 for row in rows if row.get("failed"))
    summary["runs"] = float(len(rows))
    summary["failed"] = float(failed)
    # the averages above skip NaN, so a run that crashed cannot drag them down. That is
    # only honest next to the success rate: 1 success out of 100 would otherwise be
    # reported as the score of a single lucky case.
    summary["success_rate"] = _ratio(len(rows) - failed, len(rows))
    return summary


def _ratio(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else float("nan")


def _paired_deltas(rows: Sequence[dict[str, Any]], key: str) -> dict[tuple[str, str], list[float]]:
    """Score minus the baseline score of the *same* background+product pair.

    A raw condition mean mixes in how hard each eye is; the paired delta does not.
    """
    baselines: dict[tuple[str, str], float] = {}
    for row in rows:
        if row.get("condition") == "baseline":
            value = _value(row, key)
            if value is not None:
                baselines[(str(row.get("pair_key")), f"{row['mode']}/{row['brush']}")] = value
    deltas: dict[tuple[str, str], list[float]] = {}
    for row in rows:
        if row.get("condition") == "baseline":
            continue
        base = baselines.get((str(row.get("pair_key")), f"{row['mode']}/{row['brush']}"))
        value = _value(row, key)
        if base is None or value is None:
            continue
        deltas.setdefault((str(row.get("condition")), str(row.get("condition_value"))), []).append(
            value - base
        )
    return deltas


def _value(row: dict[str, Any], key: str) -> float | None:
    raw = row.get(key)
    if not isinstance(raw, int | float) or np.isnan(float(raw)):
        return None
    return float(raw)


def _condition_key(row: dict[str, Any]) -> str:
    """`rotation_deg = 10.0`, not just `rotation_deg`.

    Averaging -10, -5, +5 and +10 degrees into one row hides exactly the thing the
    condition breakdown is supposed to show: where the score starts to fall.
    """
    condition = str(row.get("condition", "unknown"))
    value = row.get("condition_value")
    return condition if value in (None, "None", "") else f"{condition} = {value}"


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
    rows = [{**row, "condition_key": _condition_key(row)} for row in rows]
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
            for name, group in _group(rows, "condition_key").items()
        },
        "paired_delta": {
            f"{condition} = {value}": {
                "dice_delta": float(np.mean(values)),
                "pairs": len(values),
            }
            for (condition, value), values in sorted(_paired_deltas(auto, "dice").items())
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
                "success",
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
                    values["success_rate"],
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
        "`RGB MAE` only covers pixels opaque in both the prediction and the ground truth;"
        " `opaque coverage` says how much of the product that is, so a high score on a"
        " sliver cannot pass for fidelity. `recompose MAE` compares against the ground"
        " truth product composited on the same base, not against `worn.png`.",
        "",
        _table(
            [
                "mode / brush",
                "RGB MAE",
                "RGB RMSE",
                "opaque coverage",
                "recompose MAE",
                "reconstruction err",
                "sec/run",
            ],
            [
                [
                    name,
                    values["rgb_mae"],
                    values["rgb_rmse"],
                    values["opaque_coverage"],
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
        "Raw means. They still carry the difficulty of whichever eyes ended up in each"
        " group — read the paired table below before ranking conditions.",
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
    if summary.get("paired_delta"):
        parts += [
            "",
            "### Paired against the baseline of the same eye (bare & worn-only, no brush)",
            "",
            "Every condition is generated for the *same* background and product as its"
            " baseline, so this delta isolates the condition. Negative = worse than that"
            " eye's own baseline.",
            "",
            _table(
                ["condition", "Dice delta", "pairs"],
                [
                    [name, values["dice_delta"], values["pairs"]]
                    for name, values in sorted(
                        summary["paired_delta"].items(), key=lambda item: item[1]["dice_delta"]
                    )
                ],
            ),
        ]
    if summary.get("pixel_mutation"):
        parts += [
            "",
            "## Pixel mutation (product RGB under a transform)",
            "",
            "`production_default` marks the interpolation the shipped recompose path uses."
            " `fringe RGB MAE` is an **upper bound**: transparent pixels are filled with the"
            " background colour here, whereas `estimate_foreground_ml` leaves something much"
            " closer to the lash colour (docs/benchmark-findings.md §5.1).",
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
            "Measured against an INTER_NEAREST resize, which keeps original pixel values by"
            " construction. **Not a quality comparison**: nearest is not the right answer"
            " for a downscale, so `difference from nearest` counts how many product pixels"
            " stop being original pixels — it does not say the image looks worse.",
            "",
            _table(
                [
                    "target ROI width",
                    "scale",
                    "interpolation",
                    "pixels kept exactly",
                    "difference vs nearest",
                ],
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


def _json_safe(value: Any) -> Any:
    """NaN/Infinity are not valid JSON: any strict parser (browsers, jq) rejects them."""
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def write_report(output_root: str, rows: Sequence[dict[str, Any]], summary: dict[str, Any]) -> None:
    os.makedirs(output_root, exist_ok=True)
    with open(os.path.join(output_root, "summary.json"), "w", encoding="utf-8") as handle:
        json.dump(_json_safe(summary), handle, ensure_ascii=False, indent=2, default=str, allow_nan=False)
    write_csv(os.path.join(output_root, "summary.csv"), rows)
    with open(os.path.join(output_root, "report.md"), "w", encoding="utf-8") as handle:
        handle.write(render_markdown(summary))
