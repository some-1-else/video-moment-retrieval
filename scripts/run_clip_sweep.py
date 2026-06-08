"""Run a controlled CLIP retrieval sweep on a fixed Charades-STA subset."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import shutil
import sys
from typing import Sequence

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.data.charades_sta_loader import find_charades_video, parse_charades_sta_line
from src.data.qvhighlights import MomentRetrievalSample
from src.pipeline.clip_dataset_retrieval import run_clip_dataset_retrieval


DEFAULT_CONFIGS = (
    {"config_name": "clip_w8_s4_mean", "window_size": 8.0, "stride": 4.0, "aggregation": "mean"},
    {"config_name": "clip_w16_s8_mean", "window_size": 16.0, "stride": 8.0, "aggregation": "mean"},
    {"config_name": "clip_w32_s16_mean", "window_size": 32.0, "stride": 16.0, "aggregation": "mean"},
    {"config_name": "clip_w16_s8_max", "window_size": 16.0, "stride": 8.0, "aggregation": "max"},
)

SUMMARY_FIELDS = [
    "config_name",
    "window_size",
    "stride",
    "aggregation",
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


def load_samples_from_selection_manifest(
    selection_manifest: str | Path,
) -> tuple[list[MomentRetrievalSample], list[dict]]:
    """Load the fixed Charades-STA subset selected by the baseline run."""

    manifest_path = Path(selection_manifest)
    with manifest_path.open("r", encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))

    if not rows:
        raise ValueError("selection_manifest must contain at least one selected row.")

    samples: list[MomentRetrievalSample] = []
    for row_index, row in enumerate(rows):
        try:
            sample = MomentRetrievalSample(
                sample_id=str(row["sample_id"]),
                video_id=str(row["video_id"]),
                query=str(row["query"]),
                duration=float(row["duration_sec"]),
                relevant_windows=[
                    [float(row["clipped_start"]), float(row["clipped_end"])]
                ],
            )
        except KeyError as error:
            raise ValueError(
                "selection_manifest is missing a required Charades-STA selection field."
            ) from error

        if sample.duration <= 0:
            raise ValueError(f"Invalid duration in selection row {row_index}.")
        window = sample.relevant_windows[0]
        if window[1] <= window[0]:
            raise ValueError(f"Invalid clipped window in selection row {row_index}.")

        samples.append(sample)

    return samples, rows


def select_valid_charades_sta_samples(
    annotations_path: str | Path,
    videos_dir: str | Path,
    min_samples: int,
    max_samples: int,
) -> tuple[list[MomentRetrievalSample], list[dict], int]:
    """Select valid Charades-STA rows and clip GT windows to video duration."""

    if min_samples <= 0:
        raise ValueError("min_samples must be a positive integer.")
    if max_samples < min_samples:
        raise ValueError("max_samples must be greater than or equal to min_samples.")

    videos_dir = Path(videos_dir)
    rows: list[dict] = []
    samples: list[MomentRetrievalSample] = []
    clipped_windows = 0

    for row_index, line in enumerate(
        Path(annotations_path).read_text(encoding="utf-8").splitlines()
    ):
        if len(samples) >= max_samples:
            break
        if not line.strip():
            continue

        parsed = parse_charades_sta_line(line, row_index)
        video_id = str(parsed["video_id"])
        video_path = find_charades_video(videos_dir, video_id)
        if video_path is None:
            continue

        metadata = read_video_metadata(video_path)
        if not metadata["can_open"] or not metadata["can_read_first_frame"]:
            continue

        duration = metadata["duration_sec"]
        if duration is None or duration <= 0:
            continue

        raw_start = float(parsed["start_time"])
        raw_end = float(parsed["end_time"])
        clipped_start = min(max(raw_start, 0.0), duration)
        clipped_end = min(max(raw_end, 0.0), duration)
        was_clipped = clipped_start != raw_start or clipped_end != raw_end
        if clipped_end <= clipped_start:
            continue

        if was_clipped:
            clipped_windows += 1

        sample_id = f"charades_sta_sweep_{row_index}"
        samples.append(
            MomentRetrievalSample(
                sample_id=sample_id,
                video_id=video_id,
                query=str(parsed["query"]),
                duration=duration,
                relevant_windows=[[clipped_start, clipped_end]],
            )
        )
        rows.append(
            {
                "sample_id": sample_id,
                "row_index": row_index,
                "video_id": video_id,
                "query": str(parsed["query"]),
                "video_path": str(video_path),
                "duration_sec": duration,
                "raw_start": raw_start,
                "raw_end": raw_end,
                "clipped_start": clipped_start,
                "clipped_end": clipped_end,
                "was_clipped": was_clipped,
                "fps": metadata["fps"],
                "frame_count": metadata["frame_count"],
            }
        )

    if len(samples) < min_samples:
        raise ValueError(
            f"Only found {len(samples)} valid Charades-STA samples; "
            f"need at least {min_samples}."
        )

    return samples, rows, clipped_windows


def read_video_metadata(video_path: str | Path) -> dict:
    cv2 = _import_cv2()
    capture = cv2.VideoCapture(str(video_path))
    can_open = bool(capture.isOpened())
    fps = 0.0
    frame_count = 0.0
    duration_sec = None
    can_read_first_frame = False

    try:
        if can_open:
            fps = float(capture.get(cv2.CAP_PROP_FPS))
            frame_count = float(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            duration_sec = frame_count / fps if fps > 0 and frame_count > 0 else None
            can_read_first_frame = bool(capture.read()[0])
    finally:
        capture.release()

    return {
        "can_open": can_open,
        "fps": fps,
        "frame_count": frame_count,
        "duration_sec": duration_sec,
        "can_read_first_frame": can_read_first_frame,
    }


def write_selection_manifest(path: str | Path, rows: list[dict]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "sample_id",
        "row_index",
        "video_id",
        "query",
        "video_path",
        "duration_sec",
        "raw_start",
        "raw_end",
        "clipped_start",
        "clipped_end",
        "was_clipped",
        "fps",
        "frame_count",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: str | Path, value: dict) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def build_summary_row(config: dict, result: dict, unique_videos: int) -> dict:
    metrics = result["metrics"]
    stats = result["stats"]
    return {
        "config_name": config["config_name"],
        "window_size": config["window_size"],
        "stride": config["stride"],
        "aggregation": config["aggregation"],
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


def write_summary_csv(path: str | Path, rows: list[dict]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def select_configs(config_name: str | None) -> tuple[dict, ...]:
    if config_name is None:
        return DEFAULT_CONFIGS

    configs = tuple(
        config for config in DEFAULT_CONFIGS if config["config_name"] == config_name
    )
    if not configs:
        available = ", ".join(config["config_name"] for config in DEFAULT_CONFIGS)
        raise ValueError(
            f"Unknown config_name '{config_name}'. Available configs: {available}."
        )
    return configs


def run_from_args(args: argparse.Namespace) -> list[dict]:
    configs_to_run = select_configs(args.config_name)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    fixed_selection_manifest = args.output_dir / "selection_manifest.csv"

    if args.annotations_path is not None:
        samples, selection_rows, clipped_windows = select_valid_charades_sta_samples(
            annotations_path=args.annotations_path,
            videos_dir=args.videos_dir,
            min_samples=args.min_queries,
            max_samples=args.max_queries,
        )
        write_selection_manifest(fixed_selection_manifest, selection_rows)
    else:
        if args.selection_manifest is None:
            raise ValueError(
                "Either --annotations-path or --selection-manifest must be provided."
            )
        if Path(args.selection_manifest).resolve() != fixed_selection_manifest.resolve():
            shutil.copyfile(args.selection_manifest, fixed_selection_manifest)
        samples, selection_rows = load_samples_from_selection_manifest(
            fixed_selection_manifest
        )
        clipped_windows = sum(
            str(row.get("was_clipped", "")).strip().lower() == "true"
            for row in selection_rows
        )

    unique_videos = len({sample.video_id for sample in samples})

    summary_rows: list[dict] = []
    for config in configs_to_run:
        config_name = config["config_name"]
        config_dir = args.output_dir / config_name
        cache_dir = args.output_dir / "embeddings"
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
        result["charades_sta_sweep"] = {
            "selection_manifest": str(fixed_selection_manifest),
            "selected_rows": len(selection_rows),
            "unique_videos": unique_videos,
            "clipped_windows": clipped_windows,
            "config_name": config_name,
        }

        summary_row = build_summary_row(config, result, unique_videos)
        summary_rows.append(summary_row)

        write_json(config_dir / "result.json", result)
        write_json(config_dir / "metrics.json", result["metrics"])
        write_json(
            config_dir / "run_config.json",
            {
                "config_name": config_name,
                "config": result["config"],
                "charades_sta_sweep": result["charades_sta_sweep"],
                "summary": summary_row,
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

    write_summary_csv(args.output_dir / "summary.csv", summary_rows)
    return summary_rows


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a controlled CLIP sweep on a fixed Charades-STA subset."
    )
    parser.add_argument(
        "--selection-manifest",
        type=Path,
        default=None,
        help=(
            "Existing fixed subset manifest. Ignored when --annotations-path is set. "
            "Required when --annotations-path is omitted."
        ),
    )
    parser.add_argument(
        "--annotations-path",
        type=Path,
        default=None,
        help="Optional Charades-STA annotations file used to create a new fixed subset.",
    )
    parser.add_argument(
        "--videos-dir",
        type=Path,
        default=Path("data/raw/charades/videos"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory for the sweep results.",
    )
    parser.add_argument("--fps", type=float, default=1.0)
    parser.add_argument("--model-name", default="ViT-B/32")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--min-queries", "--min_queries", type=int, default=20)
    parser.add_argument("--max-queries", "--max_queries", type=int, default=200)
    parser.add_argument(
        "--config-name",
        default=None,
        help="Optional single config name to run instead of all default configs.",
    )
    parser.add_argument("--no-cache", action="store_true")

    return parser.parse_args(argv)


def _import_cv2():
    try:
        import cv2
    except ImportError as error:
        raise ImportError(
            "opencv-python is required for Charades-STA CLIP sweep sample selection."
        ) from error

    return cv2


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    summary_rows = run_from_args(args)

    print(f"Saved summary to: {args.output_dir / 'summary.csv'}")
    for row in summary_rows:
        print(
            "{config_name}: R@0.3={r03:.4f} R@0.5={r05:.4f} "
            "R@0.7={r07:.4f} mIoU={miou:.4f} frames={frames} "
            "time={time:.4f}s cache_hits={hits} cache_misses={misses}".format(
                config_name=row["config_name"],
                r03=row["R@1_IoU_0.3"],
                r05=row["R@1_IoU_0.5"],
                r07=row["R@1_IoU_0.7"],
                miou=row["mIoU"],
                frames=row["processed_frames"],
                time=row["inference_time_sec"],
                hits=row["cache_hits"],
                misses=row["cache_misses"],
            )
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
