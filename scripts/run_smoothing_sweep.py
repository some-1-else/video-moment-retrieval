"""Run CLIP similarity smoothing variants on a fixed Charades-STA subset."""

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

from scripts.run_clip_sweep import load_samples_from_selection_manifest
from src.pipeline.clip_dataset_retrieval import run_clip_dataset_retrieval


BASE_CONFIGS = (
    {"window_size": 8.0, "stride": 4.0, "aggregation": "mean"},
    {"window_size": 16.0, "stride": 8.0, "aggregation": "mean"},
)

SMOOTHING_VARIANTS = (
    {"name": "none", "window": None},
    {"name": "moving_average_3", "window": 3},
    {"name": "moving_average_5", "window": 5},
)

SUMMARY_FIELDS = [
    "config_name",
    "window_size",
    "stride",
    "aggregation",
    "smoothing",
    "evaluated_queries",
    "unique_videos",
    "processed_frames",
    "R@1_IoU_0.3",
    "R@1_IoU_0.5",
    "R@1_IoU_0.7",
    "mIoU",
    "inference_time_sec",
    "cache_hits",
    "cache_misses",
    "cache_size_bytes",
]


def run_from_args(args: argparse.Namespace) -> list[dict[str, Any]]:
    samples, selection_rows = load_samples_from_selection_manifest(args.selection_manifest)
    unique_videos = len({sample.video_id for sample in samples})
    args.output_dir.mkdir(parents=True, exist_ok=True)

    summary_rows: list[dict[str, Any]] = []
    for base_config in BASE_CONFIGS:
        for smoothing in SMOOTHING_VARIANTS:
            config_name = build_config_name(base_config, smoothing["name"])
            config_dir = args.output_dir / config_name
            result = run_clip_dataset_retrieval(
                samples=samples,
                videos_dir=args.videos_dir,
                fps=args.fps,
                window_size=base_config["window_size"],
                stride=base_config["stride"],
                aggregation=base_config["aggregation"],
                model_name=args.model_name,
                device=args.device,
                batch_size=args.batch_size,
                embeddings_cache_dir=args.embeddings_cache_dir,
                use_cache=not args.no_cache,
                smoothing_window=smoothing["window"],
            )
            result["charades_sta_smoothing"] = {
                "selection_manifest": str(args.selection_manifest),
                "selected_rows": len(selection_rows),
                "unique_videos": unique_videos,
                "config_name": config_name,
                "smoothing": smoothing["name"],
            }
            summary_row = build_summary_row(
                config_name=config_name,
                base_config=base_config,
                smoothing_name=smoothing["name"],
                result=result,
                unique_videos=unique_videos,
            )
            summary_rows.append(summary_row)

            write_json(config_dir / "result.json", result)
            write_json(config_dir / "metrics.json", result["metrics"])
            write_json(
                config_dir / "run_config.json",
                {
                    "config_name": config_name,
                    "config": result["config"],
                    "charades_sta_smoothing": result["charades_sta_smoothing"],
                    "summary": summary_row,
                },
            )
            write_json(
                config_dir / "cache_info.json",
                {
                    "embeddings_cache_dir": str(args.embeddings_cache_dir),
                    "cache_hits": result["stats"]["cache_hits"],
                    "cache_misses": result["stats"]["cache_misses"],
                    "cache_size_bytes": result["stats"]["embedding_cache_size_bytes"],
                },
            )

    write_summary_csv(args.output_dir / "summary.csv", summary_rows)
    return summary_rows


def build_config_name(base_config: dict[str, Any], smoothing_name: str) -> str:
    return (
        f"clip_w{base_config['window_size']:g}_s{base_config['stride']:g}_"
        f"{base_config['aggregation']}_{smoothing_name}"
    )


def build_summary_row(
    config_name: str,
    base_config: dict[str, Any],
    smoothing_name: str,
    result: dict[str, Any],
    unique_videos: int,
) -> dict[str, Any]:
    metrics = result["metrics"]
    stats = result["stats"]
    return {
        "config_name": config_name,
        "window_size": base_config["window_size"],
        "stride": base_config["stride"],
        "aggregation": base_config["aggregation"],
        "smoothing": smoothing_name,
        "evaluated_queries": stats["num_samples"],
        "unique_videos": unique_videos,
        "processed_frames": stats["total_frames"],
        "R@1_IoU_0.3": metrics["R@1_IoU_0.3"],
        "R@1_IoU_0.5": metrics["R@1_IoU_0.5"],
        "R@1_IoU_0.7": metrics["R@1_IoU_0.7"],
        "mIoU": metrics["mIoU"],
        "inference_time_sec": stats["inference_time_sec"],
        "cache_hits": stats["cache_hits"],
        "cache_misses": stats["cache_misses"],
        "cache_size_bytes": stats["embedding_cache_size_bytes"],
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
        description="Run CLIP similarity smoothing variants on Charades-STA."
    )
    parser.add_argument(
        "--selection-manifest",
        type=Path,
        default=Path("results/charades_sta_sweep_1000/selection_manifest.csv"),
    )
    parser.add_argument(
        "--videos-dir",
        type=Path,
        default=Path("data/raw/charades/videos"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/charades_sta_smoothing_1000"),
    )
    parser.add_argument(
        "--embeddings-cache-dir",
        type=Path,
        default=Path(".agent_memory/results/caches/charades_sta_sweep_1000/embeddings"),
    )
    parser.add_argument("--fps", type=float, default=1.0)
    parser.add_argument("--model-name", default="ViT-B/32")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--no-cache", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    summary_rows = run_from_args(args)
    print(f"Saved summary to: {args.output_dir / 'summary.csv'}")
    for row in summary_rows:
        print(
            "{config_name}: smoothing={smoothing} R@0.3={r03:.4f} "
            "R@0.5={r05:.4f} R@0.7={r07:.4f} mIoU={miou:.4f} "
            "time={time:.4f}s cache_hits={hits} cache_misses={misses}".format(
                config_name=row["config_name"],
                smoothing=row["smoothing"],
                r03=row["R@1_IoU_0.3"],
                r05=row["R@1_IoU_0.5"],
                r07=row["R@1_IoU_0.7"],
                miou=row["mIoU"],
                time=row["inference_time_sec"],
                hits=row["cache_hits"],
                misses=row["cache_misses"],
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
