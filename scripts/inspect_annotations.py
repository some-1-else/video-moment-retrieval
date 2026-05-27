"""Inspect QVHighlights-style annotation files without loading video data."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Sequence

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.data.qvhighlights import MomentRetrievalSample, load_qvhighlights_annotations


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect QVHighlights-style annotation files."
    )
    parser.add_argument(
        "--annotations",
        required=True,
        type=Path,
        help="Path to a JSON or JSONL annotation file.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional number of first samples to inspect.",
    )

    return parser.parse_args(argv)


def summarize_annotations(
    samples: list[MomentRetrievalSample],
    preview_limit: int = 5,
) -> dict:
    """Compute a small readable summary for annotation samples."""

    if not samples:
        raise ValueError("samples must not be empty.")

    if preview_limit <= 0:
        raise ValueError("preview_limit must be a positive integer.")

    durations = [sample.duration for sample in samples]
    relevant_window_counts = [len(sample.relevant_windows) for sample in samples]

    return {
        "num_samples": len(samples),
        "preview": [
            {
                "sample_id": sample.sample_id,
                "video_id": sample.video_id,
                "query": sample.query,
                "duration": sample.duration,
                "relevant_windows": sample.relevant_windows,
            }
            for sample in samples[:preview_limit]
        ],
        "min_duration": min(durations),
        "max_duration": max(durations),
        "avg_relevant_windows_per_sample": sum(relevant_window_counts) / len(samples),
    }


def run_from_args(args: argparse.Namespace) -> dict:
    samples = load_qvhighlights_annotations(args.annotations)

    if args.limit is not None:
        if args.limit <= 0:
            raise ValueError("limit must be a positive integer.")
        samples = samples[: args.limit]

    return summarize_annotations(samples)


def print_summary(summary: dict) -> None:
    print(f"Number of loaded samples: {summary['num_samples']}")
    print(f"Min duration: {summary['min_duration']:.4f}")
    print(f"Max duration: {summary['max_duration']:.4f}")
    print(
        "Average relevant_windows per sample: "
        f"{summary['avg_relevant_windows_per_sample']:.4f}"
    )
    print("Preview:")

    for index, sample in enumerate(summary["preview"], start=1):
        print(f"{index}. sample_id: {sample['sample_id']}")
        print(f"   video_id: {sample['video_id']}")
        print(f"   query: {sample['query']}")
        print(f"   duration: {sample['duration']}")
        print(f"   relevant_windows: {sample['relevant_windows']}")


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    summary = run_from_args(args)
    print_summary(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
