import pytest

from src.evaluation.metrics import mean_iou, recall_at_iou


def test_recall_at_iou_with_multiple_thresholds() -> None:
    predictions = [
        [0.0, 4.0],
        [5.0, 9.0],
        [10.0, 12.0],
    ]
    ground_truth = [
        [[0.0, 4.0]],
        [[6.0, 10.0]],
        [[13.0, 15.0]],
    ]

    assert recall_at_iou(predictions, ground_truth, threshold=0.3) == pytest.approx(2.0 / 3.0)
    assert recall_at_iou(predictions, ground_truth, threshold=0.5) == pytest.approx(2.0 / 3.0)
    assert recall_at_iou(predictions, ground_truth, threshold=0.7) == pytest.approx(1.0 / 3.0)


def test_mean_iou_uses_best_matching_ground_truth_window() -> None:
    predictions = [
        [0.0, 4.0],
        [5.0, 9.0],
    ]
    ground_truth = [
        [[0.0, 2.0], [0.0, 4.0]],
        [[7.0, 9.0], [0.0, 1.0]],
    ]

    expected = (1.0 + 0.5) / 2.0
    assert mean_iou(predictions, ground_truth) == pytest.approx(expected)


def test_metrics_raise_for_mismatched_lengths() -> None:
    predictions = [[0.0, 1.0]]
    ground_truth = []

    with pytest.raises(ValueError, match="same number of examples"):
        recall_at_iou(predictions, ground_truth, threshold=0.5)

    with pytest.raises(ValueError, match="same number of examples"):
        mean_iou(predictions, ground_truth)


def test_recall_at_iou_raises_for_invalid_prediction_window() -> None:
    predictions = [[3.0, 1.0]]
    ground_truth = [[[0.0, 2.0]]]

    with pytest.raises(ValueError, match="end >= start"):
        recall_at_iou(predictions, ground_truth, threshold=0.5)


def test_mean_iou_raises_for_invalid_ground_truth_window() -> None:
    predictions = [[0.0, 2.0]]
    ground_truth = [[[2.0, 1.0]]]

    with pytest.raises(ValueError, match="end >= start"):
        mean_iou(predictions, ground_truth)
