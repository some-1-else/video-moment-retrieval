"""Window scoring utilities for retrieval."""

from __future__ import annotations


def select_best_window(
    windows: list[list[float]],
    scores: list[float],
) -> list[float]:
    """Return the first window with the maximum score."""

    if not windows:
        raise ValueError("windows must not be empty.")

    if not scores:
        raise ValueError("scores must not be empty.")

    if len(windows) != len(scores):
        raise ValueError("windows and scores must have the same length.")

    best_index = max(range(len(scores)), key=lambda index: scores[index])
    return windows[best_index]
