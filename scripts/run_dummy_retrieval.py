"""Run deterministic dummy retrieval on a QVHighlights-style annotation file."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Sequence

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.data.qvhighlights import load_qvhighlights_annotations
from src.pipeline.dummy_dataset_retrieval import run_dummy_dataset_retrieval


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run dummy Video Moment Retrieval on annotation files."
    )
    parser.add_argument(
        "--annotations",
        required=True,
        type=Path,
        help="Path to a JSON or JSONL annotation file.",
    )
    parser.add_argument(
        "--window-size",
        required=True,
        type=float,
        help="Temporal window size in seconds.",
    )
    parser.add_argument(
        "--stride",
        required=True,
        type=float,
        help="Temporal window stride in seconds.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional number of first samples to evaluate.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path where the JSON result should be saved.",
    )

    return parser.parse_args(argv)


def run_dummy_retrieval_cli(args: argparse.Namespace) -> dict:
    samples = load_qvhighlights_annotations(args.annotations)

    if args.limit is not None:
        if args.limit <= 0:
            raise ValueError("limit must be a positive integer.")
        samples = samples[: args.limit]

    result = run_dummy_dataset_retrieval(
        samples=samples,
        window_size=args.window_size,
        stride=args.stride,
    )

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", encoding="utf-8") as output_file:
            json.dump(result, output_file, indent=2, ensure_ascii=False)

    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    result = run_dummy_retrieval_cli(args)

    print("Metrics:")
    for metric_name, value in result["metrics"].items():
        print(f"{metric_name}: {value:.4f}")

    if args.output is not None:
        print(f"Saved result to: {args.output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
