"""Lightweight frame timestamp sampling utilities."""

from __future__ import annotations


def generate_frame_timestamps(
    duration: float,
    fps: float,
) -> list[float]:
    """Generate frame timestamps inside [0, duration)."""

    num_frames = estimate_num_frames(duration, fps)

    return [frame_index / fps for frame_index in range(num_frames)]


def estimate_num_frames(
    duration: float,
    fps: float,
) -> int:
    """Estimate the number of timestamps generated for duration and FPS."""

    _validate_positive(duration, "duration")
    _validate_positive(fps, "fps")

    frame_count = 0
    while frame_count / fps < duration:
        frame_count += 1

    return frame_count


def _validate_positive(value: float, name: str) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be a positive number.")
