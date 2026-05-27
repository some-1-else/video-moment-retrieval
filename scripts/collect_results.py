"""Collect experiment result JSON files into a CSV summary table."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Sequence


CSV_COLUMNS = [
    "result_file",
    "num_samples",
    "fps",
    "window_size",
    "stride",
    "aggregation",
    "smoothing_window",
    "R@1_IoU_0.3",
    "R@1_IoU_0.5",
    "R@1_IoU_0.7",
    "mIoU",
    "total_frames",
    "avg_frames_per_sample",
    "inference_time_sec",
    "cache_hits",
    "cache_misses",
    "embedding_cache_size_bytes",
]


def collect_result_rows(results_dir: Path) -> list[dict]:
    """Read result JSON files and flatten config, metrics and stats into rows."""

    rows: list[dict] = []

    for result_path in sorted(results_dir.glob("*.json")):
        result = json.loads(result_path.read_text(encoding="utf-8"))
        config = _as_dict(result.get("config"))
        metrics = _as_dict(result.get("metrics"))
        stats = _as_dict(result.get("stats"))

        rows.append(
            {
                "result_file": result_path.name,
                "num_samples": _first_present(stats, config, "num_samples"),
                "fps": config.get("fps", ""),
                "window_size": config.get("window_size", ""),
                "stride": config.get("stride", ""),
                "aggregation": config.get("aggregation", ""),
                "smoothing_window": config.get("smoothing_window", ""),
                "R@1_IoU_0.3": metrics.get("R@1_IoU_0.3", ""),
                "R@1_IoU_0.5": metrics.get("R@1_IoU_0.5", ""),
                "R@1_IoU_0.7": metrics.get("R@1_IoU_0.7", ""),
                "mIoU": metrics.get("mIoU", ""),
                "total_frames": stats.get("total_frames", ""),
                "avg_frames_per_sample": stats.get("avg_frames_per_sample", ""),
                "inference_time_sec": stats.get("inference_time_sec", ""),
                "cache_hits": stats.get("cache_hits", ""),
                "cache_misses": stats.get("cache_misses", ""),
                "embedding_cache_size_bytes": stats.get(
                    "embedding_cache_size_bytes",
                    "",
                ),
            }
        )

    return rows


def write_summary_csv(rows: list[dict], output_path: Path) -> None:
    """Write flattened result rows to a CSV file."""

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in CSV_COLUMNS})


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect experiment result JSON files into a CSV summary."
    )
    parser.add_argument(
        "--results-dir",
        required=True,
        type=Path,
        help="Directory with JSON result files.",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Path where the CSV summary should be saved.",
    )

    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    rows = collect_result_rows(args.results_dir)
    write_summary_csv(rows, args.output)
    print(f"Collected {len(rows)} result files into: {args.output}")
    return 0


def _as_dict(value: Any) -> dict:
    if isinstance(value, dict):
        return value

    return {}


def _first_present(primary: dict, secondary: dict, key: str):
    if key in primary:
        return primary[key]

    return secondary.get(key, "")


if __name__ == "__main__":
    raise SystemExit(main())
