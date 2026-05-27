import json
from pathlib import Path

from scripts.run_qvhighlights_available_clip_retrieval import (
    filter_manifest_samples_by_validation_report,
    get_readable_video_ids,
    parse_args as parse_retrieval_args,
)
from scripts.validate_qvhighlights_videos import (
    build_validation_report,
    validate_video_file,
)


def test_validate_video_file_reports_missing_file(tmp_path) -> None:
    missing_path = tmp_path / "missing.mp4"

    result = validate_video_file(missing_path)

    assert result["exists"] is False
    assert result["readable"] is False
    assert result["duration"] is None
    assert result["frame_count"] is None
    assert "does not exist" in result["error"]


def test_build_validation_report_contains_expected_structure(tmp_path) -> None:
    existing_path = tmp_path / "not_video.mp4"
    existing_path.write_bytes(b"not a video")
    missing_path = tmp_path / "missing.mp4"
    manifest = {
        "samples": [
            {
                "video_id": "not_video",
                "video_path": str(existing_path),
                "video_exists": True,
            },
            {
                "video_id": "missing",
                "video_path": str(missing_path),
                "video_exists": True,
            },
            {
                "video_id": "not_checked",
                "video_path": str(tmp_path / "not_checked.mp4"),
                "video_exists": False,
            },
        ]
    }

    report = build_validation_report(manifest, tmp_path / "manifest.json")

    assert report["manifest"] == str(tmp_path / "manifest.json")
    assert report["num_checked"] == 2
    assert report["num_readable"] == 0
    assert report["num_unreadable"] == 2
    assert [video["video_id"] for video in report["videos"]] == [
        "not_video",
        "missing",
    ]
    assert set(report["videos"][0]) == {
        "video_id",
        "video_path",
        "exists",
        "readable",
        "duration",
        "frame_count",
        "error",
    }


def test_readable_video_filtering_uses_validation_report() -> None:
    manifest_samples = [
        {"video_id": "readable"},
        {"video_id": "unreadable"},
        {"video_id": "missing_from_report"},
    ]
    validation_report = {
        "videos": [
            {"video_id": "readable", "readable": True},
            {"video_id": "unreadable", "readable": False},
        ]
    }

    assert get_readable_video_ids(validation_report) == {"readable"}
    assert filter_manifest_samples_by_validation_report(
        manifest_samples,
        validation_report,
    ) == [{"video_id": "readable"}]


def test_run_qvhighlights_available_clip_retrieval_parse_validation_report_arg() -> None:
    args = parse_retrieval_args(
        [
            "--manifest",
            "data/qvhighlights/qvhighlights_val_diverse50_subset_manifest.json",
            "--validation-report",
            "data/qvhighlights/qvhighlights_val_diverse50_video_validation.json",
            "--fps",
            "1",
            "--window-size",
            "16",
            "--stride",
            "8",
            "--aggregation",
            "mean",
            "--model-name",
            "ViT-B/32",
            "--output",
            "results/qvhighlights_available_clip_retrieval_diverse50_w16_s8_mean.json",
        ]
    )

    assert args.manifest == Path(
        "data/qvhighlights/qvhighlights_val_diverse50_subset_manifest.json"
    )
    assert args.validation_report == Path(
        "data/qvhighlights/qvhighlights_val_diverse50_video_validation.json"
    )
    assert args.window_size == 16.0
    assert args.stride == 8.0
