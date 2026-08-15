#!/usr/bin/env python3
"""Generate a synthetic benchmark dataset.

Procedural backgrounds and a procedural product need no external data at all:

    python scripts/generate_benchmark.py --cases 100

Any local folder of photos can be used as backgrounds instead (never committed):

    python scripts/generate_benchmark.py \
        --background-dir /data/periorbital/images \
        --product evaluation-data/products/product_lash.png \
        --cases 100 --output evaluation-data/generated
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation import backgrounds as background_sources  # noqa: E402
from evaluation.generator import build_case, plan_cases, write_case  # noqa: E402
from evaluation.products import ImageProduct, ProceduralProduct, load_product_png  # noqa: E402

DEFAULT_OUTPUT = "evaluation-data/generated"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--cases", type=int, default=100, help="number of cases to generate")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="dataset directory to write")
    parser.add_argument("--background-dir", default=None, help="folder of bare eye photos (optional)")
    parser.add_argument("--backgrounds", type=int, default=12, help="procedural backgrounds to synthesise")
    parser.add_argument("--width", type=int, default=320, help="procedural background width")
    parser.add_argument("--height", type=int, default=240, help="procedural background height")
    parser.add_argument("--product", default=None, help="RGBA product PNG (optional)")
    parser.add_argument("--product-seeds", type=int, default=1, help="procedural products to synthesise")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--own-lash", type=float, default=1.0, help="strength of the model's own lashes")
    parser.add_argument(
        "--on-no-face",
        choices=("skip", "fallback"),
        default="skip",
        help="what to do with photos MediaPipe cannot detect a face in",
    )
    parser.add_argument("--clean", action="store_true", help="delete the output directory first")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.background_dir:
        sources = list(background_sources.image_backgrounds(args.background_dir, on_no_face=args.on_no_face))
        if not sources:
            print(
                f"no usable backgrounds in {args.background_dir}: MediaPipe found no face in any image.\n"
                "Eye close-ups and profiles are outside the detector's range (docs/handover.md §8);\n"
                "use full-face photos, pass --on-no-face fallback, or drop --background-dir to\n"
                "synthesise backgrounds instead.",
                file=sys.stderr,
            )
            return 2
    else:
        sources = background_sources.procedural_backgrounds(
            args.backgrounds, args.width, args.height, seed=args.seed, own_lash=args.own_lash
        )

    if args.product:
        products = [ImageProduct(load_product_png(args.product), name=os.path.basename(args.product))]
    else:
        products = [ProceduralProduct(seed=args.seed + i) for i in range(max(1, args.product_seeds))]

    by_name = {background.name: background for background in sources}
    products_by_name = {product.name: product for product in products}
    specs = plan_cases(list(by_name), list(products_by_name), count=args.cases, seed=args.seed)

    if args.clean and os.path.isdir(args.output):
        shutil.rmtree(args.output)
    os.makedirs(args.output, exist_ok=True)
    for spec in specs:
        case = build_case(by_name[spec.background], products_by_name[spec.product], spec)
        write_case(args.output, case)
        print(f"{spec.case_id}  {spec.background}  {spec.condition}={spec.condition_value}")
    print(
        f"\n{len(specs)} cases written to {args.output} "
        f"({len(by_name)} backgrounds x {len(products_by_name)} products)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
