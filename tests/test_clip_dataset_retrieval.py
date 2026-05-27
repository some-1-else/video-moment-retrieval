from pathlib import Path

import pytest

from src.pipeline.clip_dataset_retrieval import build_result, resolve_video_path


def test_resolve_video_path_returns_expected_mp4_path(tmp_path) -> None:
    videos_dir = tmp_path / "videos"
    videos_dir.mkdir()
    video_path = videos_dir / "sample_video.mp4"
    video_path.write_bytes(b"fake")

    assert resolve_video_path(videos_dir, "sample_video") == video_path


def test_resolve_video_path_raises_for_missing_file(tmp_path) -> None:
    with pytest.raises(ValueError, match="Missing video file"):
        resolve_video_path(tmp_path, "missing")


def test_clip_dataset_build_result_returns_expected_schema() -> None:
    result = build_result(
        fps=1.0,
        window_size=2.0,
        stride=1.0,
        aggregation="mean",
        model_name="ViT-B/32",
        device="cpu",
        batch_size=32,
        videos_dir=Path("videos"),
        predictions={"sample_1": [1.0, 3.0]},
        ground_truth={"sample_1": [[1.0, 3.0]]},
        prediction_windows=[[1.0, 3.0]],
        ground_truth_windows=[[[1.0, 3.0]]],
        total_frames=5,
        inference_time_sec=1.25,
    )

    assert set(result) == {"config", "predictions", "ground_truth", "metrics", "stats"}
    assert result["config"]["videos_dir"] == "videos"
    assert result["config"]["model_name"] == "ViT-B/32"
    assert result["metrics"]["R@1_IoU_0.5"] == pytest.approx(1.0)
    assert result["stats"] == {
        "num_samples": 1,
        "total_frames": 5,
        "avg_frames_per_sample": 5.0,
        "inference_time_sec": 1.25,
        "cache_hits": 0,
        "cache_misses": 0,
        "embedding_cache_size_bytes": 0,
    }
