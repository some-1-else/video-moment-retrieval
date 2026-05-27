"""Create a tiny synthetic video for local smoke tests."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a tiny synthetic MP4 video.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/sample/synthetic_video.mp4"),
        help="Output video path.",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=10.0,
        help="Video FPS.",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=5.0,
        help="Video duration in seconds.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=160,
        help="Frame width.",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=120,
        help="Frame height.",
    )

    return parser.parse_args(argv)


def create_synthetic_video(
    output_path: Path,
    fps: float = 10.0,
    duration: float = 5.0,
    width: int = 160,
    height: int = 120,
) -> None:
    if fps <= 0:
        raise ValueError("fps must be a positive number.")
    if duration <= 0:
        raise ValueError("duration must be a positive number.")
    if width <= 0 or height <= 0:
        raise ValueError("width and height must be positive integers.")

    import cv2
    import numpy as np

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    if not writer.isOpened():
        raise ValueError(f"Could not open video writer for: {output_path}")

    num_frames = int(round(duration * fps))
    try:
        for frame_index in range(num_frames):
            frame = np.zeros((height, width, 3), dtype=np.uint8)
            frame[:, :, 0] = (frame_index * 5) % 255
            frame[:, :, 1] = 60
            frame[:, :, 2] = 160

            rect_width = 32
            rect_height = 24
            max_x = max(1, width - rect_width)
            x = int((frame_index / max(1, num_frames - 1)) * max_x)
            y = height // 2 - rect_height // 2
            cv2.rectangle(
                frame,
                (x, y),
                (x + rect_width, y + rect_height),
                (40, 230, 40),
                thickness=-1,
            )
            writer.write(frame)
    finally:
        writer.release()


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    create_synthetic_video(
        output_path=args.output,
        fps=args.fps,
        duration=args.duration,
        width=args.width,
        height=args.height,
    )
    print(f"Saved synthetic video to: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
