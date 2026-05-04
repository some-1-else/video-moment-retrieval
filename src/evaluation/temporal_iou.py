"""Utilities for temporal IoU computation."""

from __future__ import annotations

from typing import Sequence


Window = Sequence[float]


def _validate_window(window: Window) -> None:
    """Validate a temporal window in [start, end] format."""

    if len(window) != 2:
        raise ValueError("Temporal window must contain exactly two values: [start, end].")

    start, end = window
    if end < start:
        raise ValueError("Temporal window must satisfy end >= start.")


def temporal_iou(pred_window: Window, gt_window: Window) -> float:
    """Compute IoU for two temporal windows."""

    _validate_window(pred_window)
    _validate_window(gt_window)

    pred_start, pred_end = pred_window
    gt_start, gt_end = gt_window

    intersection = max(0.0, min(pred_end, gt_end) - max(pred_start, gt_start))
    union = max(pred_end, gt_end) - min(pred_start, gt_start)

    if union == 0:
        return 0.0

    return intersection / union


def max_temporal_iou(pred_window: Window, gt_windows: Sequence[Window]) -> float:
    """Return the maximum temporal IoU over multiple ground-truth windows."""

    _validate_window(pred_window)

    if not gt_windows:
        return 0.0

    return max(temporal_iou(pred_window, gt_window) for gt_window in gt_windows)

