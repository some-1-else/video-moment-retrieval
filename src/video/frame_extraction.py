"""Minimal local video frame extraction utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.video.frame_sampling import generate_frame_timestamps


def extract_frames_at_timestamps(
    video_path: str | Path,
    timestamps: list[float],
) -> list:
    """Extract frames from a local video at given timestamps in seconds."""

    path = _validate_video_path(video_path)
    _validate_timestamps(timestamps)
    cv2 = _import_cv2()

    duration = get_video_duration(path)
    for timestamp in timestamps:
        if timestamp >= duration:
            raise ValueError(
                f"timestamp {timestamp} is outside video duration {duration}."
            )

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError(f"Could not open video file: {path}")

    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    video_fps = capture.get(cv2.CAP_PROP_FPS)
    if frame_count <= 0 or video_fps <= 0:
        capture.release()
        raise ValueError(f"Could not read frame metadata for video file: {path}")

    frames = []
    try:
        for timestamp in timestamps:
            frame_index = min(int(round(timestamp * video_fps)), frame_count - 1)
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            success, frame = capture.read()
            if not success:
                raise ValueError(f"Could not read frame at timestamp {timestamp}.")
            frames.append(frame)
    finally:
        capture.release()

    return frames


def extract_sampled_frames(
    video_path: str | Path,
    fps: float,
) -> tuple[list[float], list]:
    """Generate timestamps for a local video and extract frames at those timestamps."""

    if fps <= 0:
        raise ValueError("fps must be a positive number.")

    duration = get_video_duration(video_path)
    timestamps = generate_frame_timestamps(duration, fps)
    sampled_timestamps = timestamps
    while sampled_timestamps:
        try:
            frames = extract_frames_at_timestamps(video_path, sampled_timestamps)
            return sampled_timestamps, frames
        except ValueError as error:
            unreadable_timestamp = _parse_unreadable_timestamp_error(str(error))
            if unreadable_timestamp is None:
                raise

            failing_index = next(
                (
                    index
                    for index, timestamp in enumerate(sampled_timestamps)
                    if timestamp >= unreadable_timestamp
                ),
                len(sampled_timestamps) - 1,
            )
            sampled_timestamps = sampled_timestamps[:failing_index]

    raise ValueError(f"Could not extract sampled frames from video file: {video_path}")


def get_video_duration(video_path: str | Path) -> float:
    """Estimate local video duration in seconds using OpenCV metadata."""

    path = _validate_video_path(video_path)
    cv2 = _import_cv2()

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError(f"Could not open video file: {path}")

    try:
        frame_count = capture.get(cv2.CAP_PROP_FRAME_COUNT)
        fps = capture.get(cv2.CAP_PROP_FPS)
    finally:
        capture.release()

    if frame_count <= 0 or fps <= 0:
        raise ValueError(f"Could not estimate duration for video file: {path}")

    return frame_count / fps


def _validate_video_path(video_path: str | Path) -> Path:
    path = Path(video_path)
    if not path.exists():
        raise ValueError(f"video_path does not exist: {path}")

    if not path.is_file():
        raise ValueError(f"video_path must be a file: {path}")

    return path


def _validate_timestamps(timestamps: list[float]) -> None:
    for timestamp in timestamps:
        if timestamp < 0:
            raise ValueError("timestamps must not contain negative values.")


def _parse_unreadable_timestamp_error(message: str) -> float | None:
    prefix = "Could not read frame at timestamp "
    suffix = "."
    if not message.startswith(prefix) or not message.endswith(suffix):
        return None

    raw_timestamp = message[len(prefix) : -len(suffix)]
    try:
        return float(raw_timestamp)
    except ValueError:
        return None


def _import_cv2() -> Any:
    try:
        import cv2
    except ImportError as error:
        raise ImportError(
            "opencv-python is required for video frame extraction. "
            "Install project requirements before running video inspection."
        ) from error

    return cv2
