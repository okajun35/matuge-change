#!/usr/bin/env python3
"""Run the extraction pipeline over a benchmark dataset and write a report.

    python scripts/run_evaluation.py \
        --dataset evaluation-data/generated \
        --output evaluation-results

Nothing in the production pipeline is modified: the runner drives `SessionService` on a
temporary store, exactly as the HTTP API does.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation import mutation, report  # noqa: E402
from evaluation.backgrounds import procedural_backgrounds  # noqa: E402
from evaluation.dataset import load_dataset  # noqa: E402
from evaluation.products import ProceduralProduct  # noqa: E402
from evaluation.runner import BRUSHES, MODES, RunConfig, run_dataset  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--dataset", default="evaluation-data/generated")
    parser.add_argument("--output", default="evaluation-results")
    parser.add_argument("--limit", type=int, default=None, help="evaluate only the first N cases")
    parser.add_argument("--modes", nargs="+", choices=MODES, default=list(MODES))
    parser.add_argument("--brushes", nargs="+", choices=BRUSHES, default=list(BRUSHES))
    parser.add_argument("--fg-thresh", type=float, default=0.70)
    parser.add_argument("--bg-thresh", type=float, default=0.18)
    parser.add_argument("--unknown-band-px", type=int, default=6)
    parser.add_argument("--boundary-tolerance", type=int, default=2)
    parser.add_argument("--no-images", action="store_true", help="skip per-case PNG output")
    parser.add_argument("--no-mutation", action="store_true", help="skip the interpolation experiment")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    cases = load_dataset(args.dataset)
    if args.limit:
        cases = cases[: args.limit]
    config = RunConfig(
        modes=tuple(args.modes),
        brushes=tuple(args.brushes),
        fg_thresh=args.fg_thresh,
        bg_thresh=args.bg_thresh,
        unknown_band_px=args.unknown_band_px,
        boundary_tolerance=args.boundary_tolerance,
        save_images=not args.no_images,
        save_comparison=not args.no_images,
    )
    os.makedirs(args.output, exist_ok=True)
    started = time.perf_counter()
    rows = run_dataset(cases, config, args.output, progress=lambda message: print(message, flush=True))

    mutation_rows: list[dict[str, object]] = []
    roi_rows: list[dict[str, object]] = []
    if not args.no_mutation:
        print("pixel mutation experiment ...", flush=True)
        background = procedural_backgrounds(1, seed=9001)[0]
        mutation_rows = mutation.run_experiment(background, ProceduralProduct(seed=9002))
        worn = cases[0].worn
        roi_rows = mutation.roi_downscale_experiment(worn, widths=(worn.shape[1], 240, 160, 100))

    summary = report.summarise(
        rows,
        mutation_rows,
        roi_rows,
        config={
            "dataset": args.dataset,
            "cases": len(cases),
            "modes": list(config.modes),
            "brushes": list(config.brushes),
            "fg_thresh": config.fg_thresh,
            "bg_thresh": config.bg_thresh,
            "unknown_band_px": config.unknown_band_px,
            "binary_threshold": 0.5,
            "boundary_tolerance": config.boundary_tolerance,
            "wall_clock_seconds": round(time.perf_counter() - started, 1),
        },
    )
    report.write_report(args.output, rows, summary)
    print(f"\n{len(rows)} runs over {len(cases)} cases in {time.perf_counter() - started:.1f}s")
    print(f"wrote {args.output}/report.md, summary.json, summary.csv")
    for name, values in summary["overall"].items():
        print(f"  {name:>18}: Dice={values['dice']:.4f} IoU={values['iou']:.4f} MAD={values['mad']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
