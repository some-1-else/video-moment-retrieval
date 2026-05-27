import pytest

from src.retrieval.scoring import select_best_window


def test_select_best_window_returns_window_with_maximum_score() -> None:
    windows = [
        [0.0, 4.0],
        [2.0, 6.0],
        [4.0, 8.0],
    ]
    scores = [0.1, 0.8, 0.3]

    assert select_best_window(windows, scores) == [2.0, 6.0]


def test_select_best_window_returns_first_window_for_tied_maximum_score() -> None:
    windows = [
        [0.0, 4.0],
        [2.0, 6.0],
        [4.0, 8.0],
    ]
    scores = [0.8, 0.8, 0.3]

    assert select_best_window(windows, scores) == [0.0, 4.0]


def test_select_best_window_raises_for_empty_windows() -> None:
    with pytest.raises(ValueError, match="windows must not be empty"):
        select_best_window([], [0.1])


def test_select_best_window_raises_for_empty_scores() -> None:
    with pytest.raises(ValueError, match="scores must not be empty"):
        select_best_window([[0.0, 4.0]], [])


def test_select_best_window_raises_for_mismatched_lengths() -> None:
    windows = [
        [0.0, 4.0],
        [2.0, 6.0],
    ]
    scores = [0.1]

    with pytest.raises(ValueError, match="same length"):
        select_best_window(windows, scores)
