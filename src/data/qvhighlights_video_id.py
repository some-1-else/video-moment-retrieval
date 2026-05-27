"""Utilities for parsing QVHighlights video identifiers."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class QVHighlightsVideoId:
    raw_video_id: str
    youtube_id: str
    clip_start: float | None
    clip_end: float | None


def parse_qvhighlights_video_id(video_id: str) -> QVHighlightsVideoId:
    parts = video_id.split("_")
    if len(parts) < 3:
        return QVHighlightsVideoId(
            raw_video_id=video_id,
            youtube_id=video_id,
            clip_start=None,
            clip_end=None,
        )

    try:
        clip_start = float(parts[-2])
        clip_end = float(parts[-1])
    except ValueError:
        return QVHighlightsVideoId(
            raw_video_id=video_id,
            youtube_id=video_id,
            clip_start=None,
            clip_end=None,
        )

    youtube_id = "_".join(parts[:-2])
    return QVHighlightsVideoId(
        raw_video_id=video_id,
        youtube_id=youtube_id,
        clip_start=clip_start,
        clip_end=clip_end,
    )
