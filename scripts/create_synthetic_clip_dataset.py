"""Create a tiny synthetic video dataset for CLIP retrieval smoke tests."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence


SAMPLES = [
    {
        "sample_id": "synthetic_red_square",
        "video_id": "synthetic_red_square",
        "query": "red square",
        "duration": 5.0,
        "relevant_windows": [[1.0, 3.0]],
        "color_bgr": (0, 0, 255),
    },
    {
        "sample_id": "synthetic_blue_square",
        "video_id": "synthetic_blue_square",
        "query": "blue square",
        "duration": 5.0,
        "relevant_windows": [[2.0, 4.0]],
        "color_bgr": (255, 0, 0),
    },
    {
        "sample_id": "synthetic_green_square",
        "video_id": "synthetic_green_square",
        "query": "green square",
        "duration": 5.0,
        "relevant_windows": [[0.0, 2.0]],
        "color_bgr": (0, 255, 0),
    },
]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create tiny synthetic CLIP videos.")
    parser.add_argument(
        "--videos-dir",
        type=Path,
        default=Path("data/sample/synthetic_clip_videos"),
        help="Directory where synthetic MP4 videos should be written.",
    )
    parser.add_argument(
        "--annotations",
        type=Path,
        default=Path("data/sample/synthetic_clip_annotations.jsonl"),
        help="Output JSONL annotation path.",
    )
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--width", type=int, default=160)
    parser.add_argument("--height", type=int, default=120)

    return parser.parse_args(argv)


def create_synthetic_clip_dataset(
    videos_dir: Path,
    annotations_path: Path,
    fps: float = 10.0,
    width: int = 160,
    height: int = 120,
) -> None:
    videos_dir.mkdir(parents=True, exist_ok=True)
    annotations_path.parent.mkdir(parents=True, exist_ok=True)

    for sample in SAMPLES:
        create_colored_square_video(
            output_path=videos_dir / f"{sample['video_id']}.mp4",
            color_bgr=sample["color_bgr"],
            active_window=sample["relevant_windows"][0],
            duration=sample["duration"],
            fps=fps,
            width=width,
            height=height,
        )

    with annotations_path.open("w", encoding="utf-8") as output_file:
        for sample in SAMPLES:
            record = {
                "sample_id": sample["sample_id"],
                "video_id": sample["video_id"],
                "query": sample["query"],
                "duration": sample["duration"],
                "relevant_windows": sample["relevant_windows"],
            }
            output_file.write(json.dumps(record, ensure_ascii=False) + "\n")


def create_colored_square_video(
    output_path: Path,
    color_bgr: tuple[int, int, int],
    active_window: list[float],
    duration: float,
    fps: float,
    width: int,
    height: int,
) -> None:
    if fps <= 0:
        raise ValueError("fps must be a positive number.")

    import cv2
    import numpy as np

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    if not writer.isOpened():
        raise ValueError(f"Could not open video writer for: {output_path}")

    num_frames = int(round(duration * fps))
    start_time, end_time = active_window
    square_size = 44
    y = height // 2 - square_size // 2
    max_x = max(1, width - square_size)

    try:
        for frame_index in range(num_frames):
            timestamp = frame_index / fps
            frame = np.full((height, width, 3), 38, dtype=np.uint8)

            if start_time <= timestamp <= end_time:
                progress = (timestamp - start_time) / max(0.001, end_time - start_time)
                x = int(progress * max_x)
                cv2.rectangle(
                    frame,
                    (x, y),
                    (x + square_size, y + square_size),
                    color_bgr,
                    thickness=-1,
                )

            writer.write(frame)
    finally:
        writer.release()


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    create_synthetic_clip_dataset(
        videos_dir=args.videos_dir,
        annotations_path=args.annotations,
        fps=args.fps,
        width=args.width,
        height=args.height,
    )
    print(f"Saved synthetic videos to: {args.videos_dir}")
    print(f"Saved annotations to: {args.annotations}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
