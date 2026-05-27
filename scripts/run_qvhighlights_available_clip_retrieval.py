"""Run CLIP retrieval only on locally available QVHighlights subset videos."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Sequence

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.data.qvhighlights import MomentRetrievalSample
from src.pipeline.clip_dataset_retrieval import run_clip_dataset_retrieval


def load_manifest(manifest_path: str | Path) -> dict:
    return json.loads(Path(manifest_path).read_text(encoding="utf-8"))


def get_available_manifest_samples(manifest: dict) -> list[dict]:
    return [
        sample
        for sample in manifest.get("samples", [])
        if sample.get("video_exists") is True
    ]


def load_validation_report(validation_report_path: str | Path) -> dict:
    return json.loads(Path(validation_report_path).read_text(encoding="utf-8"))


def get_readable_video_ids(validation_report: dict) -> set[str]:
    return {
        str(video.get("video_id", ""))
        for video in validation_report.get("videos", [])
        if video.get("readable") is True
    }


def filter_manifest_samples_by_validation_report(
    manifest_samples: list[dict],
    validation_report: dict,
) -> list[dict]:
    readable_video_ids = get_readable_video_ids(validation_report)
    return [
        sample
        for sample in manifest_samples
        if sample.get("video_id") in readable_video_ids
    ]


def build_moment_samples_from_manifest_entries(
    manifest_samples: list[dict],
) -> list[MomentRetrievalSample]:
    if not manifest_samples:
        raise ValueError(
            "No available videos found in manifest. Download a small subset first "
            "or recreate the manifest after adding local videos."
        )

    return [
        MomentRetrievalSample(
            sample_id=str(sample["sample_id"]),
            video_id=str(sample["video_id"]),
            query=str(sample["query"]),
            duration=float(sample["duration"]),
            relevant_windows=[
                [float(window[0]), float(window[1])]
                for window in sample["relevant_windows"]
            ],
        )
        for sample in manifest_samples
    ]


def resolve_videos_dir_from_manifest(manifest: dict, available_samples: list[dict]) -> Path:
    if manifest.get("videos_dir"):
        return Path(manifest["videos_dir"])

    if not available_samples:
        raise ValueError("Cannot resolve videos_dir because no available samples exist.")

    video_path = available_samples[0].get("video_path")
    if not video_path:
        raise ValueError("Manifest does not contain videos_dir or sample video_path.")

    return Path(video_path).parent


def run_from_args(args: argparse.Namespace) -> dict:
    manifest = load_manifest(args.manifest)
    available_entries = get_available_manifest_samples(manifest)
    if args.validation_report is not None:
        validation_report = load_validation_report(args.validation_report)
        available_entries = filter_manifest_samples_by_validation_report(
            available_entries,
            validation_report,
        )
        if not available_entries:
            raise ValueError(
                "No readable videos found after applying validation report. "
                "Validate local videos or download a readable subset first."
            )

    samples = build_moment_samples_from_manifest_entries(available_entries)
    videos_dir = resolve_videos_dir_from_manifest(manifest, available_entries)

    result = run_clip_dataset_retrieval(
        samples=samples,
        videos_dir=videos_dir,
        fps=args.fps,
        window_size=args.window_size,
        stride=args.stride,
        aggregation=args.aggregation,
        model_name=args.model_name,
        device=args.device,
        batch_size=args.batch_size,
        embeddings_cache_dir=args.embeddings_cache_dir,
        use_cache=args.use_cache,
    )

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(result, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    return result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run CLIP retrieval on locally available QVHighlights subset videos."
    )
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--validation-report", type=Path, default=None)
    parser.add_argument("--fps", required=True, type=float)
    parser.add_argument("--window-size", required=True, type=float)
    parser.add_argument("--stride", required=True, type=float)
    parser.add_argument("--aggregation", choices=("mean", "max"), default="mean")
    parser.add_argument("--model-name", default="ViT-B/32")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--embeddings-cache-dir", type=Path, default=None)
    parser.add_argument("--use-cache", action="store_true")
    parser.add_argument("--output", type=Path, default=None)

    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        result = run_from_args(args)
    except ValueError as exc:
        print(f"Cannot run QVHighlights available-subset CLIP retrieval: {exc}")
        return 1

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
