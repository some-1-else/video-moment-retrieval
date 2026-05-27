import pytest

from src.video.frame_sampling import estimate_num_frames, generate_frame_timestamps


def test_generate_frame_timestamps_with_fps_one() -> None:
    assert generate_frame_timestamps(duration=5.0, fps=1.0) == pytest.approx(
        [0.0, 1.0, 2.0, 3.0, 4.0]
    )


def test_generate_frame_timestamps_with_fps_two() -> None:
    assert generate_frame_timestamps(duration=5.0, fps=2.0) == pytest.approx(
        [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5]
    )


def test_generate_frame_timestamps_with_non_integer_duration() -> None:
    assert generate_frame_timestamps(duration=2.2, fps=1.0) == pytest.approx(
        [0.0, 1.0, 2.0]
    )


def test_estimate_num_frames_matches_generated_timestamps() -> None:
    duration = 4.5
    fps = 2.0

    timestamps = generate_frame_timestamps(duration=duration, fps=fps)

    assert estimate_num_frames(duration=duration, fps=fps) == len(timestamps)


def test_last_frame_timestamp_is_strictly_less_than_duration() -> None:
    duration = 2.2
    timestamps = generate_frame_timestamps(duration=duration, fps=2.0)

    assert timestamps[-1] < duration


def test_generate_frame_timestamps_raises_for_non_positive_duration() -> None:
    with pytest.raises(ValueError, match="duration must be a positive number"):
        generate_frame_timestamps(duration=0.0, fps=1.0)


def test_generate_frame_timestamps_raises_for_non_positive_fps() -> None:
    with pytest.raises(ValueError, match="fps must be a positive number"):
        generate_frame_timestamps(duration=5.0, fps=0.0)
