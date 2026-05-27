"""Dataset-level dummy retrieval pipeline for sanity-checking experiments."""

from __future__ import annotations

from src.data.qvhighlights import MomentRetrievalSample
from src.evaluation.metrics import mean_iou, recall_at_iou
from src.retrieval.scoring import select_best_window
from src.retrieval.windowing import generate_temporal_windows


def run_dummy_dataset_retrieval(
    samples: list[MomentRetrievalSample],
    window_size: float,
    stride: float,
) -> dict:
    """Run deterministic dummy retrieval over multiple annotated samples."""

    if not samples:
        raise ValueError("samples must not be empty.")

    if window_size <= 0:
        raise ValueError("window_size must be a positive number.")

    if stride <= 0:
        raise ValueError("stride must be a positive number.")

    predictions: dict[str, list[float]] = {}
    ground_truth: dict[str, list[list[float]]] = {}
    prediction_windows: list[list[float]] = []
    ground_truth_windows: list[list[list[float]]] = []

    for sample in samples:
        windows = generate_temporal_windows(sample.duration, window_size, stride)
        scores = _score_windows_by_video_center(windows, sample.duration)
        prediction = select_best_window(windows, scores)

        predictions[sample.sample_id] = prediction
        ground_truth[sample.sample_id] = sample.relevant_windows
        prediction_windows.append(prediction)
        ground_truth_windows.append(sample.relevant_windows)

    return {
        "config": {
            "window_size": float(window_size),
            "stride": float(stride),
            "num_samples": len(samples),
        },
        "predictions": predictions,
        "ground_truth": ground_truth,
        "metrics": {
            "R@1_IoU_0.3": recall_at_iou(
                prediction_windows,
                ground_truth_windows,
                threshold=0.3,
            ),
            "R@1_IoU_0.5": recall_at_iou(
                prediction_windows,
                ground_truth_windows,
                threshold=0.5,
            ),
            "R@1_IoU_0.7": recall_at_iou(
                prediction_windows,
                ground_truth_windows,
                threshold=0.7,
            ),
            "mIoU": mean_iou(prediction_windows, ground_truth_windows),
        },
    }


def _score_windows_by_video_center(windows: list[list[float]], duration: float) -> list[float]:
    video_center = duration / 2.0

    return [
        -abs(((window[0] + window[1]) / 2.0) - video_center)
        for window in windows
    ]
