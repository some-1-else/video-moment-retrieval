import json
from pathlib import Path

from scripts.download_qvhighlights_subset_videos import (
    build_download_plan,
    download_subset_videos,
    get_missing_samples,
    load_manifest,
    parse_args,
)


def _manifest() -> dict:
    return {
        "samples": [
            {
                "sample_id": "sample_available",
                "video_id": "available_0.0_10.0",
                "youtube_id": "available",
                "clip_start": 0.0,
                "clip_end": 10.0,
                "video_exists": True,
            },
            {
                "sample_id": "sample_missing_1",
                "video_id": "missing_10.0_20.0",
                "youtube_id": "missing",
                "clip_start": 10.0,
                "clip_end": 20.0,
                "video_exists": False,
            },
            {
                "sample_id": "sample_missing_2",
                "video_id": "missing_with_underscore_20.0_30.0",
                "youtube_id": "missing_with_underscore",
                "clip_start": 20.0,
                "clip_end": 30.0,
                "video_exists": False,
            },
        ]
    }


def test_load_manifest_reads_json(tmp_path) -> None:
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(_manifest()), encoding="utf-8")

    assert load_manifest(manifest_path) == _manifest()


def test_get_missing_samples_selects_only_missing_videos() -> None:
    missing_samples = get_missing_samples(_manifest())

    assert [sample["sample_id"] for sample in missing_samples] == [
        "sample_missing_1",
        "sample_missing_2",
    ]


def test_build_download_plan_respects_max_downloads_and_output_paths(tmp_path) -> None:
    plan = build_download_plan(_manifest(), tmp_path / "videos", max_downloads=1)

    assert plan == [
        {
            "sample_id": "sample_missing_1",
            "video_id": "missing_10.0_20.0",
            "youtube_id": "missing",
            "clip_start": 10.0,
            "clip_end": 20.0,
            "youtube_url": "https://www.youtube.com/watch?v=missing",
            "output_path": str(tmp_path / "videos" / "missing_10.0_20.0.mp4"),
        }
    ]


def test_download_subset_videos_dry_run_does_not_create_files(tmp_path) -> None:
    summary = download_subset_videos(
        manifest=_manifest(),
        output_dir=tmp_path / "videos",
        max_downloads=2,
        dry_run=True,
    )

    assert summary["attempted"] == 0
    assert summary["downloaded"] == 0
    assert summary["skipped_existing"] == 0
    assert summary["failed"] == 0
    assert len(summary["plan"]) == 2
    assert not (tmp_path / "videos").exists()


def test_download_subset_videos_skips_existing_output_without_network(tmp_path) -> None:
    output_dir = tmp_path / "videos"
    output_dir.mkdir()
    existing_path = output_dir / "missing_10.0_20.0.mp4"
    existing_path.write_bytes(b"existing")

    summary = download_subset_videos(
        manifest=_manifest(),
        output_dir=output_dir,
        max_downloads=1,
        dry_run=False,
    )

    assert summary["attempted"] == 0
    assert summary["downloaded"] == 0
    assert summary["skipped_existing"] == 1
    assert summary["failed"] == 0


def test_download_qvhighlights_subset_videos_parse_args() -> None:
    args = parse_args(
        [
            "--manifest",
            "data/qvhighlights/qvhighlights_val_subset_manifest.json",
            "--output-dir",
            "data/qvhighlights/videos",
            "--max-downloads",
            "3",
            "--dry-run",
        ]
    )

    assert args.manifest == Path("data/qvhighlights/qvhighlights_val_subset_manifest.json")
    assert args.output_dir == Path("data/qvhighlights/videos")
    assert args.max_downloads == 3
    assert args.dry_run is True
