import pytest

from src.retrieval.windowing import generate_temporal_windows


def test_generate_temporal_windows_with_exact_last_window() -> None:
    windows = generate_temporal_windows(duration=10.0, window_size=4.0, stride=2.0)

    assert windows == [
        [0.0, 4.0],
        [2.0, 6.0],
        [4.0, 8.0],
        [6.0, 10.0],
    ]


def test_generate_temporal_windows_adds_final_window_ending_at_duration() -> None:
    windows = generate_temporal_windows(duration=10.0, window_size=4.0, stride=3.0)

    assert windows == [
        [0.0, 4.0],
        [3.0, 7.0],
        [6.0, 10.0],
    ]


def test_generate_temporal_windows_returns_single_window_when_window_size_covers_duration() -> None:
    assert generate_temporal_windows(duration=10.0, window_size=10.0, stride=2.0) == [
        [0.0, 10.0]
    ]
    assert generate_temporal_windows(duration=10.0, window_size=12.0, stride=2.0) == [
        [0.0, 10.0]
    ]


def test_generate_temporal_windows_stay_inside_duration_bounds() -> None:
    duration = 10.0
    windows = generate_temporal_windows(duration=duration, window_size=4.0, stride=3.0)

    assert all(0.0 <= start <= end <= duration for start, end in windows)


def test_generate_temporal_windows_raises_for_non_positive_duration() -> None:
    with pytest.raises(ValueError, match="duration must be a positive number"):
        generate_temporal_windows(duration=0.0, window_size=4.0, stride=2.0)


def test_generate_temporal_windows_raises_for_non_positive_window_size() -> None:
    with pytest.raises(ValueError, match="window_size must be a positive number"):
        generate_temporal_windows(duration=10.0, window_size=0.0, stride=2.0)


def test_generate_temporal_windows_raises_for_non_positive_stride() -> None:
    with pytest.raises(ValueError, match="stride must be a positive number"):
        generate_temporal_windows(duration=10.0, window_size=4.0, stride=0.0)
