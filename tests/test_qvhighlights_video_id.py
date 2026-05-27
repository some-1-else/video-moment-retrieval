from src.data.qvhighlights_video_id import (
    QVHighlightsVideoId,
    parse_qvhighlights_video_id,
)


def test_parse_qvhighlights_video_id_standard_format() -> None:
    parsed = parse_qvhighlights_video_id("NUsG9BgSes0_210.0_360.0")

    assert parsed == QVHighlightsVideoId(
        raw_video_id="NUsG9BgSes0_210.0_360.0",
        youtube_id="NUsG9BgSes0",
        clip_start=210.0,
        clip_end=360.0,
    )


def test_parse_qvhighlights_video_id_with_underscores() -> None:
    parsed = parse_qvhighlights_video_id("abc_def_10.0_20.0")

    assert parsed.youtube_id == "abc_def"
    assert parsed.clip_start == 10.0
    assert parsed.clip_end == 20.0


def test_parse_qvhighlights_video_id_plain_id() -> None:
    parsed = parse_qvhighlights_video_id("plain_video_id")

    assert parsed == QVHighlightsVideoId(
        raw_video_id="plain_video_id",
        youtube_id="plain_video_id",
        clip_start=None,
        clip_end=None,
    )


def test_parse_qvhighlights_video_id_malformed_timestamps() -> None:
    parsed = parse_qvhighlights_video_id("abc_def_start_end")

    assert parsed.youtube_id == "abc_def_start_end"
    assert parsed.clip_start is None
    assert parsed.clip_end is None


def test_parse_qvhighlights_video_id_keeps_non_increasing_range() -> None:
    parsed = parse_qvhighlights_video_id("abc_20.0_10.0")

    assert parsed.youtube_id == "abc"
    assert parsed.clip_start == 20.0
    assert parsed.clip_end == 10.0
