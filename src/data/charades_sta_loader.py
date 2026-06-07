"""Utilities for reading Charades-STA temporal grounding annotations."""

from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Any

LOGGER = logging.getLogger(__name__)

VIDEO_EXTENSIONS = (".mp4", ".avi", ".mkv")


def parse_charades_sta_line(line: str, index: int = 0) -> dict[str, Any]:
    """Parse one Charades-STA annotation line.

    Expected format:
        video_id start_time end_time##sentence
    """

    stripped_line = line.strip()
    if not stripped_line:
        raise ValueError(f"Charades-STA annotation line {index} must not be empty.")

    if "##" not in stripped_line:
        raise ValueError(
            f"Charades-STA annotation line {index} must contain '##' before the query."
        )

    metadata, query = stripped_line.split("##", maxsplit=1)
    metadata_parts = metadata.split()
    if len(metadata_parts) != 3:
        raise ValueError(
            f"Charades-STA annotation line {index} must have format "
            "'video_id start_time end_time##query'."
        )

    video_id, raw_start_time, raw_end_time = metadata_parts
    try:
        start_time = float(raw_start_time)
        end_time = float(raw_end_time)
    except ValueError as error:
        raise ValueError(
            f"Charades-STA annotation line {index} has non-numeric start/end times."
        ) from error

    if end_time < start_time:
        raise ValueError(
            f"Charades-STA annotation line {index} must satisfy end_time >= start_time."
        )

    query = query.strip()
    if not query:
        raise ValueError(f"Charades-STA annotation line {index} has an empty query.")

    return {
        "video_id": video_id,
        "start_time": start_time,
        "end_time": end_time,
        "query": query,
    }


def load_charades_sta_annotations(
    annotations_path: str | Path,
    split: str,
    videos_dir: str | Path | None = None,
    metadata_csv_path: str | Path | None = None,
) -> list[dict[str, Any]]:
    """Load Charades-STA annotations in the internal retrieval record format."""

    annotations_path = Path(annotations_path)
    duration_by_video_id = (
        load_charades_duration_csv(metadata_csv_path) if metadata_csv_path else {}
    )
    samples: list[dict[str, Any]] = []

    for row_index, line in enumerate(annotations_path.read_text(encoding="utf-8").splitlines()):
        if not line.strip():
            continue

        parsed = parse_charades_sta_line(line, row_index)
        video_id = parsed["video_id"]
        duration = duration_by_video_id.get(video_id)
        if duration is None and videos_dir is not None:
            duration = get_video_duration_if_available(Path(videos_dir), video_id)
        if duration is None:
            LOGGER.warning(
                "Duration is unavailable for Charades-STA video_id '%s' at row %s.",
                video_id,
                row_index,
            )

        samples.append(
            {
                "qid": f"charades_sta_{split}_{row_index}",
                "vid": video_id,
                "query": parsed["query"],
                "duration": duration,
                "relevant_windows": [[parsed["start_time"], parsed["end_time"]]],
                "source_dataset": "charades_sta",
            }
        )

    return samples


def load_charades_duration_csv(path: str | Path) -> dict[str, float]:
    """Load video durations from a Charades metadata CSV when available."""

    csv_path = Path(path)
    with csv_path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames is None:
            return {}

        video_id_field = _find_field(reader.fieldnames, ("id", "video_id", "vid"))
        duration_field = _find_field(
            reader.fieldnames,
            ("duration", "length", "video_duration", "duration_sec"),
        )
        if video_id_field is None or duration_field is None:
            raise ValueError(
                "Charades metadata CSV must contain a video id field "
                "and a duration field."
            )

        durations: dict[str, float] = {}
        for row_index, row in enumerate(reader, start=2):
            raw_video_id = row.get(video_id_field, "").strip()
            raw_duration = row.get(duration_field, "").strip()
            if not raw_video_id or not raw_duration:
                continue
            try:
                duration = float(raw_duration)
            except ValueError as error:
                raise ValueError(
                    f"Invalid duration in Charades metadata CSV at row {row_index}."
                ) from error
            if duration > 0:
                durations[raw_video_id] = duration

    return durations


def find_charades_video(videos_dir: str | Path, video_id: str) -> Path | None:
    """Find a local Charades video file by id and known video extensions."""

    videos_dir = Path(videos_dir)
    for extension in VIDEO_EXTENSIONS:
        video_path = videos_dir / f"{video_id}{extension}"
        if video_path.exists():
            return video_path

    return None


def get_video_duration_if_available(videos_dir: str | Path, video_id: str) -> float | None:
    """Read video duration from OpenCV metadata when the local file is available."""

    video_path = find_charades_video(videos_dir, video_id)
    if video_path is None:
        return None

    try:
        cv2 = _import_cv2()
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            return None
        try:
            fps = float(capture.get(cv2.CAP_PROP_FPS))
            frame_count = float(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            if fps <= 0 or frame_count <= 0:
                return None
            return frame_count / fps
        finally:
            capture.release()
    except Exception as exc:
        LOGGER.warning("Could not read duration for video '%s': %s", video_path, exc)
        return None


def _find_field(fieldnames: list[str], candidates: tuple[str, ...]) -> str | None:
    normalized_fields = {field.lower(): field for field in fieldnames}
    for candidate in candidates:
        if candidate in normalized_fields:
            return normalized_fields[candidate]
    return None


def _import_cv2() -> Any:
    try:
        import cv2
    except ImportError as error:
        raise ImportError(
            "opencv-python is required to read Charades video metadata. "
            "Install project requirements before loading durations from videos."
        ) from error

    return cv2
