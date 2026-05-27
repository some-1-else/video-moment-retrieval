"""Temporal window generation utilities for retrieval."""

from __future__ import annotations


def _validate_positive(value: float, name: str) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be a positive number.")


def generate_temporal_windows(
    duration: float,
    window_size: float,
    stride: float,
) -> list[list[float]]:
    """Generate temporal windows fully contained in [0, duration]."""

    _validate_positive(duration, "duration")
    _validate_positive(window_size, "window_size")
    _validate_positive(stride, "stride")

    if window_size >= duration:
        return [[0.0, float(duration)]]

    windows: list[list[float]] = []
    start = 0.0

    while start + window_size <= duration:
        end = start + window_size
        windows.append([float(start), float(end)])
        start += stride

    last_end = windows[-1][1]
    if last_end < duration:
        windows.append([float(duration - window_size), float(duration)])

    return windows
