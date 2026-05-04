"""Evaluation metrics for video moment retrieval."""

from __future__ import annotations

from typing import Sequence

from src.evaluation.temporal_iou import Window, max_temporal_iou


def _validate_inputs(
    predictions: Sequence[Window],
    ground_truth: Sequence[Sequence[Window]],
) -> None:
    """Ensure predictions and ground truth have compatible lengths."""

    if len(predictions) != len(ground_truth):
        raise ValueError("Predictions and ground truth must have the same number of examples.")


def recall_at_iou(
    predictions: Sequence[Window],
    ground_truth: Sequence[Sequence[Window]],
    threshold: float,
) -> float:
    """Compute Recall@1 for a given IoU threshold."""

    _validate_inputs(predictions, ground_truth)

    if not 0.0 <= threshold <= 1.0:
        raise ValueError("IoU threshold must be in the range [0, 1].")

    if not predictions:
        return 0.0

    hits = sum(
        1
        for pred_window, gt_windows in zip(predictions, ground_truth)
        if max_temporal_iou(pred_window, gt_windows) >= threshold
    )

    return hits / len(predictions)


def mean_iou(
    predictions: Sequence[Window],
    ground_truth: Sequence[Sequence[Window]],
) -> float:
    """Compute mean IoU using the best-matching ground-truth window per example."""

    _validate_inputs(predictions, ground_truth)

    if not predictions:
        return 0.0

    scores = [
        max_temporal_iou(pred_window, gt_windows)
        for pred_window, gt_windows in zip(predictions, ground_truth)
    ]

    return sum(scores) / len(scores)

