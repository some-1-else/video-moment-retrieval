"""Inspect a QVHighlights local-video subset manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence


def load_manifest(manifest_path: str | Path) -> dict:
    return json.loads(Path(manifest_path).read_text(encoding="utf-8"))


def summarize_manifest(manifest: dict, preview_limit: int = 5) -> dict:
    samples = manifest.get("samples", [])
    available_samples = [
        sample for sample in samples if sample.get("video_exists") is True
    ]
    missing_samples = [
        sample for sample in samples if sample.get("video_exists") is not True
    ]
    parsed_clip_samples = [
        sample
        for sample in samples
        if sample.get("clip_start") is not None and sample.get("clip_end") is not None
    ]
    unique_youtube_ids = []
    seen_youtube_ids = set()
    for sample in samples:
        youtube_id = sample.get("youtube_id", sample.get("video_id", ""))
        if youtube_id in seen_youtube_ids:
            continue
        unique_youtube_ids.append(youtube_id)
        seen_youtube_ids.add(youtube_id)

    return {
        "strategy": manifest.get("strategy", "unknown"),
        "num_samples": manifest.get("num_samples", len(samples)),
        "num_available_videos": manifest.get(
            "num_available_videos", len(available_samples)
        ),
        "num_missing_videos": manifest.get("num_missing_videos", len(missing_samples)),
        "num_parsed_clip_ranges": len(parsed_clip_samples),
        "num_unique_youtube_ids": len(unique_youtube_ids),
        "unique_youtube_ids": unique_youtube_ids[:preview_limit],
        "missing_video_ids": [
            sample.get("video_id", "") for sample in missing_samples[:preview_limit]
        ],
        "available_video_ids": [
            sample.get("video_id", "") for sample in available_samples[:preview_limit]
        ],
        "youtube_ids": [
            sample.get("youtube_id", sample.get("video_id", ""))
            for sample in samples[:preview_limit]
        ],
        "clip_ranges": [
            [sample.get("clip_start"), sample.get("clip_end")]
            for sample in parsed_clip_samples[:preview_limit]
        ],
        "examples": [
            {
                "sample_id": sample.get("sample_id", ""),
                "video_id": sample.get("video_id", ""),
                "youtube_id": sample.get("youtube_id", sample.get("video_id", "")),
                "clip_start": sample.get("clip_start"),
                "clip_end": sample.get("clip_end"),
                "query": sample.get("query", ""),
                "relevant_windows": sample.get("relevant_windows", []),
            }
            for sample in samples[:preview_limit]
        ],
    }


def print_summary(summary: dict) -> None:
    print(f"Strategy: {summary['strategy']}")
    print(f"Samples: {summary['num_samples']}")
    print(f"Available videos: {summary['num_available_videos']}")
    print(f"Missing videos: {summary['num_missing_videos']}")
    print(f"Parsed clip ranges: {summary['num_parsed_clip_ranges']}")
    print(f"Unique YouTube IDs: {summary['num_unique_youtube_ids']}")

    print("First unique YouTube IDs:")
    for youtube_id in summary["unique_youtube_ids"]:
        print(f"- {youtube_id}")

    print("First missing video IDs:")
    for video_id in summary["missing_video_ids"]:
        print(f"- {video_id}")

    print("First available video IDs:")
    for video_id in summary["available_video_ids"]:
        print(f"- {video_id}")

    print("First parsed YouTube IDs:")
    for youtube_id in summary["youtube_ids"]:
        print(f"- {youtube_id}")

    print("First parsed clip ranges:")
    for clip_start, clip_end in summary["clip_ranges"]:
        print(f"- [{clip_start}, {clip_end}]")

    print("Sample queries:")
    for example in summary["examples"]:
        print(
            f"- {example['sample_id']} | {example['video_id']} | "
            f"{example['youtube_id']} | [{example['clip_start']}, {example['clip_end']}] | "
            f"{example['query']} | {example['relevant_windows']}"
        )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect a QVHighlights local-video subset manifest."
    )
    parser.add_argument("--manifest", required=True, type=Path)

    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    manifest = load_manifest(args.manifest)
    print_summary(summarize_manifest(manifest))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
