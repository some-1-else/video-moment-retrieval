import pytest

from src.retrieval.aggregation import aggregate_frame_scores_to_windows, smooth_scores


def test_aggregate_frame_scores_to_windows_with_mean() -> None:
    frame_timestamps = [1.0, 2.0, 5.0, 9.0]
    frame_scores = [0.2, 0.4, 0.8, 0.1]
    windows = [[0.0, 3.0], [4.0, 10.0]]

    scores = aggregate_frame_scores_to_windows(
        frame_timestamps,
        frame_scores,
        windows,
        method="mean",
    )

    assert scores == pytest.approx([0.3, 0.45])


def test_aggregate_frame_scores_to_windows_with_max() -> None:
    frame_timestamps = [1.0, 2.0, 5.0, 9.0]
    frame_scores = [0.2, 0.4, 0.8, 0.1]
    windows = [[0.0, 3.0], [4.0, 10.0]]

    scores = aggregate_frame_scores_to_windows(
        frame_timestamps,
        frame_scores,
        windows,
        method="max",
    )

    assert scores == pytest.approx([0.4, 0.8])


def test_aggregate_frame_scores_to_windows_returns_zero_for_empty_window() -> None:
    scores = aggregate_frame_scores_to_windows(
        frame_timestamps=[1.0, 2.0],
        frame_scores=[0.2, 0.4],
        windows=[[5.0, 6.0]],
        method="mean",
    )

    assert scores == pytest.approx([0.0])


def test_aggregate_frame_scores_to_windows_raises_for_mismatched_lengths() -> None:
    with pytest.raises(ValueError, match="same length"):
        aggregate_frame_scores_to_windows(
            frame_timestamps=[1.0, 2.0],
            frame_scores=[0.2],
            windows=[[0.0, 3.0]],
        )


def test_aggregate_frame_scores_to_windows_raises_for_unknown_method() -> None:
    with pytest.raises(ValueError, match="method must be one of"):
        aggregate_frame_scores_to_windows(
            frame_timestamps=[1.0],
            frame_scores=[0.2],
            windows=[[0.0, 3.0]],
            method="median",
        )


def test_smooth_scores_with_window_size_one_returns_original_scores() -> None:
    scores = [1.0, 2.0, 3.0]

    assert smooth_scores(scores, window_size=1) == scores


def test_smooth_scores_with_window_size_three_uses_centered_moving_average() -> None:
    scores = [1.0, 2.0, 3.0, 4.0, 5.0]

    assert smooth_scores(scores, window_size=3) == pytest.approx(
        [1.5, 2.0, 3.0, 4.0, 4.5]
    )


def test_smooth_scores_works_when_window_size_is_larger_than_scores() -> None:
    scores = [1.0, 2.0, 3.0]

    assert smooth_scores(scores, window_size=5) == pytest.approx([2.0, 2.0, 2.0])


def test_smooth_scores_raises_for_non_positive_window_size() -> None:
    with pytest.raises(ValueError, match="window_size must be a positive integer"):
        smooth_scores([1.0, 2.0, 3.0], window_size=0)
