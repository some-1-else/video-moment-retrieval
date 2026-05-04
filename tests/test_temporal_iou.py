import pytest

from src.evaluation.temporal_iou import max_temporal_iou, temporal_iou


def test_temporal_iou_identical_intervals() -> None:
    assert temporal_iou([1.0, 4.0], [1.0, 4.0]) == pytest.approx(1.0)


def test_temporal_iou_partial_overlap() -> None:
    assert temporal_iou([0.0, 4.0], [2.0, 6.0]) == pytest.approx(2.0 / 6.0)


def test_temporal_iou_no_overlap() -> None:
    assert temporal_iou([0.0, 2.0], [3.0, 5.0]) == pytest.approx(0.0)


def test_max_temporal_iou_uses_best_ground_truth_window() -> None:
    gt_windows = [[0.0, 1.0], [2.0, 5.0], [6.0, 8.0]]

    assert max_temporal_iou([2.5, 4.5], gt_windows) == pytest.approx(2.0 / 3.0)


def test_temporal_iou_raises_for_invalid_prediction_window() -> None:
    with pytest.raises(ValueError, match="end >= start"):
        temporal_iou([5.0, 2.0], [1.0, 3.0])


def test_max_temporal_iou_raises_for_invalid_ground_truth_window() -> None:
    with pytest.raises(ValueError, match="end >= start"):
        max_temporal_iou([1.0, 2.0], [[3.0, 1.0]])

