"""Score aggregation utilities for temporal retrieval."""

from __future__ import annotations


def aggregate_frame_scores_to_windows(
    frame_timestamps: list[float],
    frame_scores: list[float],
    windows: list[list[float]],
    method: str = "mean",
) -> list[float]:
    """Aggregate frame-level scores into temporal window scores."""

    if len(frame_timestamps) != len(frame_scores):
        raise ValueError("frame_timestamps and frame_scores must have the same length.")

    if method not in {"mean", "max"}:
        raise ValueError("method must be one of: mean, max.")

    window_scores: list[float] = []

    for window_index, window in enumerate(windows):
        _validate_window(window, window_index)
        start, end = window
        scores_inside_window = [
            score
            for timestamp, score in zip(frame_timestamps, frame_scores)
            if start <= timestamp <= end
        ]

        if not scores_inside_window:
            window_scores.append(0.0)
        elif method == "mean":
            window_scores.append(sum(scores_inside_window) / len(scores_inside_window))
        else:
            window_scores.append(max(scores_inside_window))

    return window_scores


def smooth_scores(
    scores: list[float],
    window_size: int,
) -> list[float]:
    """Smooth scores with a centered moving average."""

    if window_size <= 0:
        raise ValueError("window_size must be a positive integer.")

    if window_size <= 1:
        return list(scores)

    smoothed_scores: list[float] = []
    left_radius = window_size // 2
    right_radius = window_size - left_radius - 1

    for index in range(len(scores)):
        start = max(0, index - left_radius)
        end = min(len(scores), index + right_radius + 1)
        window_values = scores[start:end]
        smoothed_scores.append(sum(window_values) / len(window_values))

    return smoothed_scores


def _validate_window(window: list[float], window_index: int) -> None:
    if len(window) != 2:
        raise ValueError(f"Window {window_index} must have format [start, end].")

    start, end = window
    if end < start:
        raise ValueError(f"Window {window_index} must satisfy end >= start.")
