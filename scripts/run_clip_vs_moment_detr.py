"""Run CLIP configs on the same 50-row subset used by Moment-DETR and compare."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Any, Sequence

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.data.qvhighlights import load_qvhighlights_annotations
from src.pipeline.clip_dataset_retrieval import run_clip_dataset_retrieval


CLIP_CONFIGS = (
    {"config_name": "clip_w8_s4_mean", "window_size": 8.0, "stride": 4.0, "aggregation": "mean"},
    {"config_name": "clip_w16_s8_mean", "window_size": 16.0, "stride": 8.0, "aggregation": "mean"},
    {"config_name": "clip_w32_s16_mean", "window_size": 32.0, "stride": 16.0, "aggregation": "mean"},
    {"config_name": "clip_w16_s8_max", "window_size": 16.0, "stride": 8.0, "aggregation": "max"},
)

SUMMARY_FIELDS = [
    "model",
    "config_name",
    "window_size",
    "stride",
    "aggregation",
    "evaluated_queries",
    "failed_queries",
    "R@1_IoU_0.3",
    "R@1_IoU_0.5",
    "R@1_IoU_0.7",
    "mIoU",
    "inference_time_sec",
    "cache_hits",
    "cache_misses",
    "cache_size_bytes",
]


def run_from_args(args: argparse.Namespace) -> dict[str, Any]:
    samples = load_qvhighlights_annotations(args.subset_jsonl)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = args.output_dir / "embeddings"

    clip_rows = []
    comparison_rows = []
    for config in CLIP_CONFIGS:
        config_name = config["config_name"]
        config_dir = args.output_dir / config_name
        result = run_clip_dataset_retrieval(
            samples=samples,
            videos_dir=args.videos_dir,
            fps=args.fps,
            window_size=config["window_size"],
            stride=config["stride"],
            aggregation=config["aggregation"],
            model_name=args.model_name,
            device=args.device,
            batch_size=args.batch_size,
            embeddings_cache_dir=cache_dir,
            use_cache=not args.no_cache,
        )
        row = build_clip_summary_row(config, result)
        clip_rows.append(row)
        comparison_rows.append(row)

        write_json(config_dir / "result.json", result)
        write_json(config_dir / "metrics.json", result["metrics"])
        write_json(
            config_dir / "run_config.json",
            {
                "subset_jsonl": str(args.subset_jsonl),
                "config_name": config_name,
                "config": result["config"],
                "summary": row,
            },
        )
        write_json(
            config_dir / "cache_info.json",
            {
                "embeddings_cache_dir": str(cache_dir),
                "cache_hits": result["stats"]["cache_hits"],
                "cache_misses": result["stats"]["cache_misses"],
                "cache_size_bytes": result["stats"]["embedding_cache_size_bytes"],
            },
        )

    moment_row = build_moment_detr_summary_row(args.moment_detr_metrics)
    comparison_rows.append(moment_row)

    write_summary_csv(args.output_dir / "clip_summary.csv", clip_rows)
    write_summary_csv(args.output_dir / "comparison_summary.csv", comparison_rows)

    return {
        "clip_rows": clip_rows,
        "moment_detr_row": moment_row,
        "comparison_rows": comparison_rows,
    }


def build_clip_summary_row(config: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
    metrics = result["metrics"]
    stats = result["stats"]
    return {
        "model": "CLIP",
        "config_name": config["config_name"],
        "window_size": config["window_size"],
        "stride": config["stride"],
        "aggregation": config["aggregation"],
        "evaluated_queries": stats["num_samples"],
        "failed_queries": 0,
        "R@1_IoU_0.3": metrics["R@1_IoU_0.3"],
        "R@1_IoU_0.5": metrics["R@1_IoU_0.5"],
        "R@1_IoU_0.7": metrics["R@1_IoU_0.7"],
        "mIoU": metrics["mIoU"],
        "inference_time_sec": stats["inference_time_sec"],
        "cache_hits": stats["cache_hits"],
        "cache_misses": stats["cache_misses"],
        "cache_size_bytes": stats["embedding_cache_size_bytes"],
    }


def build_moment_detr_summary_row(metrics_path: str | Path) -> dict[str, Any]:
    metrics = json.loads(Path(metrics_path).read_text(encoding="utf-8"))
    return {
        "model": "Moment-DETR",
        "config_name": "moment_detr_raw_video",
        "window_size": "",
        "stride": "",
        "aggregation": "",
        "evaluated_queries": metrics["evaluated_queries"],
        "failed_queries": metrics["failed_queries"],
        "R@1_IoU_0.3": metrics["R@1_IoU_0.3"],
        "R@1_IoU_0.5": metrics["R@1_IoU_0.5"],
        "R@1_IoU_0.7": metrics["R@1_IoU_0.7"],
        "mIoU": metrics["mIoU"],
        "inference_time_sec": metrics["inference_time_sec"],
        "cache_hits": "",
        "cache_misses": "",
        "cache_size_bytes": "",
    }


def write_json(path: str | Path, data: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def write_summary_csv(path: str | Path, rows: list[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare CLIP configs against saved Moment-DETR metrics on 50 rows."
    )
    parser.add_argument(
        "--subset-jsonl",
        type=Path,
        default=Path("data/processed/charades_sta_moment_detr_test_subset.jsonl"),
    )
    parser.add_argument(
        "--moment-detr-metrics",
        type=Path,
        default=Path("results/moment_detr_charades_50/metrics.json"),
    )
    parser.add_argument(
        "--videos-dir",
        type=Path,
        default=Path("data/raw/charades/videos"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/clip_vs_moment_detr_50"),
    )
    parser.add_argument("--fps", type=float, default=1.0)
    parser.add_argument("--model-name", default="ViT-B/32")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--no-cache", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    result = run_from_args(args)
    print(f"Saved CLIP summary to: {args.output_dir / 'clip_summary.csv'}")
    print(f"Saved comparison summary to: {args.output_dir / 'comparison_summary.csv'}")
    for row in result["comparison_rows"]:
        print(
            "{model} {config}: R@0.3={r03:.4f} R@0.5={r05:.4f} "
            "R@0.7={r07:.4f} mIoU={miou:.4f} time={time:.4f}s".format(
                model=row["model"],
                config=row["config_name"],
                r03=float(row["R@1_IoU_0.3"]),
                r05=float(row["R@1_IoU_0.5"]),
                r07=float(row["R@1_IoU_0.7"]),
                miou=float(row["mIoU"]),
                time=float(row["inference_time_sec"]),
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
