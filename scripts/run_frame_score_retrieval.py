"""Run frame-score retrieval without CLIP or video decoding."""

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
from src.pipeline.frame_score_retrieval import run_frame_score_retrieval


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run deterministic frame-score retrieval on annotation files."
    )
    parser.add_argument(
        "--annotations",
        required=True,
        type=Path,
        help="Path to a JSON or JSONL annotation file.",
    )
    parser.add_argument(
        "--fps",
        required=True,
        type=float,
        help="Frame sampling rate in frames per second.",
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
        "--aggregation",
        choices=("mean", "max"),
        default="mean",
        help="How to aggregate frame scores inside each temporal window.",
    )
    parser.add_argument(
        "--smoothing-window",
        type=int,
        default=None,
        help="Optional moving average window size for frame scores.",
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


def run_from_args(args: argparse.Namespace) -> dict:
    samples = load_qvhighlights_annotations(args.annotations)

    if args.limit is not None:
        if args.limit <= 0:
            raise ValueError("limit must be a positive integer.")
        samples = samples[: args.limit]

    result = run_frame_score_retrieval(
        samples=samples,
        fps=args.fps,
        window_size=args.window_size,
        stride=args.stride,
        aggregation=args.aggregation,
        smoothing_window=args.smoothing_window,
    )

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", encoding="utf-8") as output_file:
            json.dump(result, output_file, indent=2, ensure_ascii=False)

    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    result = run_from_args(args)

    print("Metrics:")
    for metric_name, value in result["metrics"].items():
        print(f"{metric_name}: {value:.4f}")

    print("Stats:")
    for stat_name, value in result["stats"].items():
        if isinstance(value, float):
            print(f"{stat_name}: {value:.4f}")
        else:
            print(f"{stat_name}: {value}")

    if args.output is not None:
        print(f"Saved result to: {args.output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
