"""Create a local-video availability manifest for a small QVHighlights subset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Sequence

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.data.qvhighlights import MomentRetrievalSample, load_qvhighlights_annotations
from src.data.qvhighlights_video_id import parse_qvhighlights_video_id


def select_subset_samples(
    samples: list[MomentRetrievalSample],
    limit: int,
    strategy: str = "first",
) -> list[MomentRetrievalSample]:
    if limit <= 0:
        raise ValueError("limit must be a positive integer")
    if strategy not in {"first", "diverse-youtube"}:
        raise ValueError("strategy must be one of: first, diverse-youtube")

    if strategy == "first":
        return samples[:limit]

    selected_samples = []
    seen_youtube_ids = set()
    for sample in samples:
        parsed_video_id = parse_qvhighlights_video_id(sample.video_id)
        if parsed_video_id.youtube_id in seen_youtube_ids:
            continue

        selected_samples.append(sample)
        seen_youtube_ids.add(parsed_video_id.youtube_id)
        if len(selected_samples) == limit:
            break

    return selected_samples


def build_subset_manifest(
    samples: list[MomentRetrievalSample],
    annotations_path: str | Path,
    videos_dir: str | Path,
    limit: int,
    strategy: str = "first",
) -> dict:
    selected_samples = select_subset_samples(samples, limit, strategy)
    videos_dir = Path(videos_dir)
    manifest_samples = []

    for sample in selected_samples:
        video_path = videos_dir / f"{sample.video_id}.mp4"
        parsed_video_id = parse_qvhighlights_video_id(sample.video_id)
        manifest_samples.append(
            {
                "sample_id": sample.sample_id,
                "video_id": sample.video_id,
                "youtube_id": parsed_video_id.youtube_id,
                "clip_start": parsed_video_id.clip_start,
                "clip_end": parsed_video_id.clip_end,
                "query": sample.query,
                "duration": sample.duration,
                "relevant_windows": sample.relevant_windows,
                "video_path": str(video_path),
                "video_exists": video_path.exists(),
            }
        )

    num_available_videos = sum(
        1 for sample_entry in manifest_samples if sample_entry["video_exists"]
    )
    num_missing_videos = len(manifest_samples) - num_available_videos

    return {
        "annotations": str(annotations_path),
        "videos_dir": str(videos_dir),
        "limit": limit,
        "strategy": strategy,
        "num_samples": len(manifest_samples),
        "num_available_videos": num_available_videos,
        "num_missing_videos": num_missing_videos,
        "samples": manifest_samples,
    }


def create_subset_manifest(
    annotations_path: str | Path,
    videos_dir: str | Path,
    limit: int,
    strategy: str = "first",
) -> dict:
    samples = load_qvhighlights_annotations(annotations_path)
    return build_subset_manifest(
        samples,
        annotations_path,
        videos_dir,
        limit,
        strategy=strategy,
    )


def write_manifest(manifest: dict, output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a local-video availability manifest for QVHighlights samples."
    )
    parser.add_argument("--annotations", required=True, type=Path)
    parser.add_argument("--videos-dir", required=True, type=Path)
    parser.add_argument("--limit", required=True, type=int)
    parser.add_argument(
        "--strategy",
        choices=("first", "diverse-youtube"),
        default="first",
    )
    parser.add_argument("--output", required=True, type=Path)

    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    manifest = create_subset_manifest(
        annotations_path=args.annotations,
        videos_dir=args.videos_dir,
        limit=args.limit,
        strategy=args.strategy,
    )
    write_manifest(manifest, args.output)

    print(f"Saved manifest to: {args.output}")
    print(f"Strategy: {manifest['strategy']}")
    print(f"Samples: {manifest['num_samples']}")
    print(f"Available videos: {manifest['num_available_videos']}")
    print(f"Missing videos: {manifest['num_missing_videos']}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
