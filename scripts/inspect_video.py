"""Inspect local video frame extraction without running CLIP."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Sequence

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.video.frame_extraction import extract_sampled_frames, get_video_duration


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect frame extraction from a local video file."
    )
    parser.add_argument(
        "--video",
        required=True,
        type=Path,
        help="Path to a local video file.",
    )
    parser.add_argument(
        "--fps",
        required=True,
        type=float,
        help="Sampling rate in frames per second.",
    )

    return parser.parse_args(argv)


def run_from_args(args: argparse.Namespace) -> dict:
    duration = get_video_duration(args.video)
    timestamps, frames = extract_sampled_frames(args.video, args.fps)

    first_frame_shape = None
    if frames:
        first_frame_shape = tuple(frames[0].shape)

    return {
        "duration": duration,
        "num_timestamps": len(timestamps),
        "num_frames": len(frames),
        "first_frame_shape": first_frame_shape,
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    result = run_from_args(args)

    print(f"Estimated duration: {result['duration']:.4f}")
    print(f"Number of sampled timestamps: {result['num_timestamps']}")
    print(f"Number of extracted frames: {result['num_frames']}")
    print(f"First frame shape: {result['first_frame_shape']}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
