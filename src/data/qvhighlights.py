"""Utilities for reading QVHighlights-style moment retrieval annotations."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class MomentRetrievalSample:
    """Single text-to-video moment retrieval annotation."""

    sample_id: str
    video_id: str
    query: str
    duration: float
    relevant_windows: list[list[float]]


_SAMPLE_ID_FIELDS = ("sample_id", "qid", "annotation_id")
_VIDEO_ID_FIELDS = ("video_id", "vid")
_QUERY_FIELDS = ("query", "query_txt", "sentence")
_DURATION_FIELDS = ("duration", "video_duration")
_RELEVANT_WINDOWS_FIELDS = ("relevant_windows", "timestamps", "gt_windows")


def load_qvhighlights_annotations(path: str | Path) -> list[MomentRetrievalSample]:
    """Load QVHighlights-style annotations from a JSON list or JSONL file."""

    annotation_path = Path(path)
    text = annotation_path.read_text(encoding="utf-8")
    raw_records = _load_raw_records(text)

    if not raw_records:
        raise ValueError("Annotation file must contain at least one object.")

    return [_parse_sample(record, index) for index, record in enumerate(raw_records)]


def _load_raw_records(text: str) -> list[dict[str, Any]]:
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return _load_jsonl_records(text)

    if isinstance(data, list):
        records = data
    elif isinstance(data, dict):
        records = [data]
    else:
        raise ValueError("Annotation JSON must be a list of objects or JSONL objects.")

    return [_ensure_object(record, index) for index, record in enumerate(records)]


def _load_jsonl_records(text: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []

    for line_number, line in enumerate(text.splitlines(), start=1):
        stripped_line = line.strip()
        if not stripped_line:
            continue

        try:
            record = json.loads(stripped_line)
        except json.JSONDecodeError as error:
            raise ValueError(f"Invalid JSONL object at line {line_number}.") from error

        records.append(_ensure_object(record, line_number))

    return records


def _ensure_object(record: Any, index: int) -> dict[str, Any]:
    if not isinstance(record, dict):
        raise ValueError(f"Annotation record {index} must be an object.")

    return record


def _parse_sample(record: dict[str, Any], index: int) -> MomentRetrievalSample:
    sample_id = _get_required_field(record, _SAMPLE_ID_FIELDS, "sample_id", index)
    video_id = _get_required_field(record, _VIDEO_ID_FIELDS, "video_id", index)
    query = _get_required_field(record, _QUERY_FIELDS, "query", index)
    duration = _parse_duration(
        _get_required_field(record, _DURATION_FIELDS, "duration", index),
        index,
    )
    relevant_windows = _parse_relevant_windows(
        _get_required_field(record, _RELEVANT_WINDOWS_FIELDS, "relevant_windows", index),
        index,
    )

    return MomentRetrievalSample(
        sample_id=str(sample_id),
        video_id=str(video_id),
        query=str(query),
        duration=duration,
        relevant_windows=relevant_windows,
    )


def _get_required_field(
    record: dict[str, Any],
    candidates: tuple[str, ...],
    field_name: str,
    index: int,
) -> Any:
    for candidate in candidates:
        if candidate in record:
            return record[candidate]

    candidate_names = ", ".join(candidates)
    raise ValueError(
        f"Missing required field '{field_name}' in annotation record {index}. "
        f"Expected one of: {candidate_names}."
    )


def _parse_duration(value: Any, index: int) -> float:
    try:
        duration = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"duration in annotation record {index} must be a number.") from error

    if duration <= 0:
        raise ValueError(f"duration in annotation record {index} must be a positive number.")

    return duration


def _parse_relevant_windows(value: Any, index: int) -> list[list[float]]:
    if not isinstance(value, list):
        raise ValueError(f"relevant_windows in annotation record {index} must be a list.")

    if not value:
        raise ValueError(f"relevant_windows in annotation record {index} must not be empty.")

    return [_parse_window(window, index, window_index) for window_index, window in enumerate(value)]


def _parse_window(window: Any, record_index: int, window_index: int) -> list[float]:
    if not isinstance(window, list) or len(window) != 2:
        raise ValueError(
            f"Window {window_index} in annotation record {record_index} "
            "must have format [start, end]."
        )

    try:
        start = float(window[0])
        end = float(window[1])
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"Window {window_index} in annotation record {record_index} "
            "must contain numeric start and end values."
        ) from error

    if end < start:
        raise ValueError(
            f"Window {window_index} in annotation record {record_index} "
            "must satisfy end >= start."
        )

    return [start, end]
