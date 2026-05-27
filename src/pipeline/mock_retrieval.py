"""Small mock end-to-end retrieval example without CLIP or video loading."""

from __future__ import annotations

from src.evaluation.metrics import mean_iou, recall_at_iou
from src.evaluation.temporal_iou import max_temporal_iou
from src.retrieval.scoring import select_best_window
from src.retrieval.windowing import generate_temporal_windows


def run_mock_retrieval_example() -> dict:
    """Run a deterministic mock retrieval example for sanity-checking the pipeline."""

    duration = 10.0
    ground_truth = [[4.0, 8.0]]
    windows = generate_temporal_windows(duration=duration, window_size=4.0, stride=2.0)
    scores = [0.1, 0.3, 0.9, 0.2]

    prediction = select_best_window(windows, scores)
    predictions = [prediction]
    ground_truth_batch = [ground_truth]

    return {
        "duration": duration,
        "windows": windows,
        "scores": scores,
        "prediction": prediction,
        "ground_truth": ground_truth,
        "iou": max_temporal_iou(prediction, ground_truth),
        "metrics": {
            "R@1_IoU_0.3": recall_at_iou(predictions, ground_truth_batch, threshold=0.3),
            "R@1_IoU_0.5": recall_at_iou(predictions, ground_truth_batch, threshold=0.5),
            "R@1_IoU_0.7": recall_at_iou(predictions, ground_truth_batch, threshold=0.7),
            "mIoU": mean_iou(predictions, ground_truth_batch),
        },
    }
