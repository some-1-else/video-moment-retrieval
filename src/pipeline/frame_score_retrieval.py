"""Frame-score based retrieval pipeline without CLIP or video decoding."""

from __future__ import annotations

from src.data.qvhighlights import MomentRetrievalSample
from src.evaluation.metrics import mean_iou, recall_at_iou
from src.retrieval.aggregation import aggregate_frame_scores_to_windows, smooth_scores
from src.retrieval.scoring import select_best_window
from src.retrieval.windowing import generate_temporal_windows
from src.video.frame_sampling import generate_frame_timestamps


def run_frame_score_retrieval(
    samples: list[MomentRetrievalSample],
    fps: float,
    window_size: float,
    stride: float,
    aggregation: str = "mean",
    smoothing_window: int | None = None,
) -> dict:
    """Run deterministic frame-score retrieval over multiple samples."""

    if not samples:
        raise ValueError("samples must not be empty.")

    if fps <= 0:
        raise ValueError("fps must be a positive number.")

    if window_size <= 0:
        raise ValueError("window_size must be a positive number.")

    if stride <= 0:
        raise ValueError("stride must be a positive number.")

    predictions: dict[str, list[float]] = {}
    ground_truth: dict[str, list[list[float]]] = {}
    prediction_windows: list[list[float]] = []
    ground_truth_windows: list[list[list[float]]] = []
    total_frames = 0

    for sample in samples:
        frame_timestamps = generate_frame_timestamps(sample.duration, fps)
        windows = generate_temporal_windows(sample.duration, window_size, stride)
        frame_scores = _score_frames_by_video_center(frame_timestamps, sample.duration)

        if smoothing_window is not None:
            frame_scores = smooth_scores(frame_scores, smoothing_window)

        window_scores = aggregate_frame_scores_to_windows(
            frame_timestamps=frame_timestamps,
            frame_scores=frame_scores,
            windows=windows,
            method=aggregation,
        )
        prediction = select_best_window(windows, window_scores)

        predictions[sample.sample_id] = prediction
        ground_truth[sample.sample_id] = sample.relevant_windows
        prediction_windows.append(prediction)
        ground_truth_windows.append(sample.relevant_windows)
        total_frames += len(frame_timestamps)

    return {
        "config": {
            "fps": float(fps),
            "window_size": float(window_size),
            "stride": float(stride),
            "aggregation": aggregation,
            "smoothing_window": smoothing_window,
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
        "stats": {
            "num_samples": len(samples),
            "total_frames": total_frames,
            "avg_frames_per_sample": total_frames / len(samples),
        },
    }


def _score_frames_by_video_center(
    frame_timestamps: list[float],
    duration: float,
) -> list[float]:
    video_center = duration / 2.0

    return [-abs(timestamp - video_center) for timestamp in frame_timestamps]
