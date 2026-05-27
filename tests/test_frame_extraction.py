from pathlib import Path

import pytest

from src.video.frame_extraction import extract_frames_at_timestamps, extract_sampled_frames


def test_extract_frames_at_timestamps_raises_for_missing_video_file(tmp_path) -> None:
    missing_path = tmp_path / "missing.mp4"

    with pytest.raises(ValueError, match="video_path does not exist"):
        extract_frames_at_timestamps(missing_path, [0.0])


def test_extract_frames_at_timestamps_raises_for_negative_timestamp(tmp_path) -> None:
    video_path = tmp_path / "video.mp4"
    video_path.write_bytes(b"not a real video")

    with pytest.raises(ValueError, match="negative"):
        extract_frames_at_timestamps(video_path, [-1.0])


def test_extract_sampled_frames_raises_for_non_positive_fps() -> None:
    with pytest.raises(ValueError, match="fps must be a positive number"):
        extract_sampled_frames(Path("missing.mp4"), fps=0.0)
