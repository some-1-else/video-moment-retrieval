from pathlib import Path

import pytest

from scripts.run_qvhighlights_available_clip_retrieval import (
    build_moment_samples_from_manifest_entries,
    get_available_manifest_samples,
    parse_args,
    resolve_videos_dir_from_manifest,
)


def _manifest() -> dict:
    return {
        "videos_dir": "data/qvhighlights/videos",
        "samples": [
            {
                "sample_id": "sample_1",
                "video_id": "video_1_0.0_10.0",
                "query": "person opens a door",
                "duration": 10.0,
                "relevant_windows": [[2.0, 6.0]],
                "video_path": "data/qvhighlights/videos/video_1_0.0_10.0.mp4",
                "video_exists": True,
            },
            {
                "sample_id": "sample_2",
                "video_id": "video_2_10.0_20.0",
                "query": "person sits down",
                "duration": 10.0,
                "relevant_windows": [[1.0, 3.0]],
                "video_path": "data/qvhighlights/videos/video_2_10.0_20.0.mp4",
                "video_exists": False,
            },
        ],
    }


def test_get_available_manifest_samples_filters_available_only() -> None:
    available_samples = get_available_manifest_samples(_manifest())

    assert len(available_samples) == 1
    assert available_samples[0]["sample_id"] == "sample_1"


def test_build_moment_samples_from_manifest_entries() -> None:
    available_samples = get_available_manifest_samples(_manifest())
    samples = build_moment_samples_from_manifest_entries(available_samples)

    assert len(samples) == 1
    assert samples[0].sample_id == "sample_1"
    assert samples[0].video_id == "video_1_0.0_10.0"
    assert samples[0].query == "person opens a door"
    assert samples[0].duration == 10.0
    assert samples[0].relevant_windows == [[2.0, 6.0]]


def test_build_moment_samples_raises_when_no_available_videos() -> None:
    with pytest.raises(ValueError, match="No available videos"):
        build_moment_samples_from_manifest_entries([])


def test_resolve_videos_dir_from_manifest_prefers_manifest_videos_dir() -> None:
    available_samples = get_available_manifest_samples(_manifest())

    assert resolve_videos_dir_from_manifest(_manifest(), available_samples) == Path(
        "data/qvhighlights/videos"
    )


def test_resolve_videos_dir_from_manifest_can_use_video_path_parent() -> None:
    manifest = _manifest()
    manifest.pop("videos_dir")
    available_samples = get_available_manifest_samples(manifest)

    assert resolve_videos_dir_from_manifest(manifest, available_samples) == Path(
        "data/qvhighlights/videos"
    )


def test_run_qvhighlights_available_clip_retrieval_parse_args() -> None:
    args = parse_args(
        [
            "--manifest",
            "data/qvhighlights/qvhighlights_val_subset_manifest.json",
            "--fps",
            "1",
            "--window-size",
            "8",
            "--stride",
            "4",
            "--aggregation",
            "mean",
            "--model-name",
            "ViT-B/32",
            "--use-cache",
            "--embeddings-cache-dir",
            "results/embeddings/qvhighlights_readable",
            "--output",
            "results/qvhighlights_available_clip_retrieval.json",
        ]
    )

    assert args.manifest == Path("data/qvhighlights/qvhighlights_val_subset_manifest.json")
    assert args.fps == 1.0
    assert args.window_size == 8.0
    assert args.stride == 4.0
    assert args.aggregation == "mean"
    assert args.model_name == "ViT-B/32"
    assert args.use_cache is True
    assert args.embeddings_cache_dir == Path("results/embeddings/qvhighlights_readable")
    assert args.output == Path("results/qvhighlights_available_clip_retrieval.json")
