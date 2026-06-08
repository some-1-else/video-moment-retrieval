"""Build a CLIP vs Moment-DETR comparison table from saved full-test metrics."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Sequence


SUMMARY_FIELDS = [
    "model",
    "config_name",
    "evaluated_queries",
    "failed_queries",
    "R@1_IoU_0.3",
    "R@1_IoU_0.5",
    "R@1_IoU_0.7",
    "mIoU",
    "inference_time_sec",
]


def read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_clip_metrics(metrics_path: Path, run_config_path: Path) -> dict[str, Any]:
    metrics = read_json(metrics_path)
    run_config = read_json(run_config_path)
    summary = run_config.get("summary", {})
    return {
        "model": "CLIP",
        "config_name": run_config.get("config_name", summary.get("config_name", "")),
        "evaluated_queries": summary.get("evaluated_queries"),
        "failed_queries": summary.get("failed_queries", 0),
        "R@1_IoU_0.3": metrics.get("R@1_IoU_0.3"),
        "R@1_IoU_0.5": metrics.get("R@1_IoU_0.5"),
        "R@1_IoU_0.7": metrics.get("R@1_IoU_0.7"),
        "mIoU": metrics.get("mIoU"),
        "inference_time_sec": summary.get("inference_time_sec"),
    }


def load_moment_detr_metrics(metrics_path: Path) -> dict[str, Any]:
    metrics = read_json(metrics_path)
    return {
        "model": "Moment-DETR",
        "config_name": "moment_detr_raw_video",
        "evaluated_queries": metrics.get("evaluated_queries"),
        "failed_queries": metrics.get("failed_queries"),
        "R@1_IoU_0.3": metrics.get("R@1_IoU_0.3"),
        "R@1_IoU_0.5": metrics.get("R@1_IoU_0.5"),
        "R@1_IoU_0.7": metrics.get("R@1_IoU_0.7"),
        "mIoU": metrics.get("mIoU"),
        "inference_time_sec": metrics.get("inference_time_sec"),
    }


def validate_summary_row(row: dict[str, Any], expected_queries: int) -> None:
    evaluated_queries = row.get("evaluated_queries")
    if evaluated_queries != expected_queries:
        raise ValueError(
            f"{row['model']} evaluated_queries must be {expected_queries}, "
            f"found {evaluated_queries}."
        )
    missing = [field for field in SUMMARY_FIELDS if row.get(field) is None]
    if missing:
        raise ValueError(f"{row['model']} row is missing fields: {missing}.")


def write_summary_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build CLIP vs Moment-DETR comparison from saved metrics."
    )
    parser.add_argument("--clip-metrics", type=Path, required=True)
    parser.add_argument("--clip-run-config", type=Path, required=True)
    parser.add_argument("--moment-detr-metrics", type=Path, required=True)
    parser.add_argument("--moment-detr-run-config", type=Path, required=True)
    parser.add_argument("--expected-queries", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.output_dir.exists():
        raise FileExistsError(
            f"Output directory already exists and will not be overwritten: "
            f"{args.output_dir}"
        )

    clip_row = load_clip_metrics(args.clip_metrics, args.clip_run_config)
    moment_row = load_moment_detr_metrics(args.moment_detr_metrics)
    validate_summary_row(clip_row, args.expected_queries)
    validate_summary_row(moment_row, args.expected_queries)

    args.output_dir.mkdir(parents=True)
    write_summary_csv(args.output_dir / "comparison_summary.csv", [clip_row, moment_row])
    run_config = {
        "clip_metrics": str(args.clip_metrics),
        "clip_run_config": str(args.clip_run_config),
        "moment_detr_metrics": str(args.moment_detr_metrics),
        "moment_detr_run_config": str(args.moment_detr_run_config),
        "expected_queries": args.expected_queries,
        "output_dir": str(args.output_dir),
    }
    (args.output_dir / "run_config.json").write_text(
        json.dumps(run_config, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"Saved comparison to: {args.output_dir / 'comparison_summary.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
