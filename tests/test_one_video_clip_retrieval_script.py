from pathlib import Path

from scripts.run_one_video_clip_retrieval import build_result, parse_args


def test_parse_args_for_one_video_clip_retrieval() -> None:
    args = parse_args(
        [
            "--video",
            "path/to/video.mp4",
            "--query",
            "person opens a door",
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
            "--output",
            "results/one_video_clip_retrieval.json",
        ]
    )

    assert args.video == Path("path/to/video.mp4")
    assert args.query == "person opens a door"
    assert args.fps == 1.0
    assert args.window_size == 8.0
    assert args.stride == 4.0
    assert args.aggregation == "mean"
    assert args.model_name == "ViT-B/32"
    assert args.output == Path("results/one_video_clip_retrieval.json")


def test_build_result_returns_expected_structure() -> None:
    result = build_result(
        video_path="video.mp4",
        query="person opens a door",
        fps=1.0,
        window_size=8.0,
        stride=4.0,
        aggregation="mean",
        model_name="ViT-B/32",
        device="cpu",
        duration=12.0,
        frame_timestamps=[0.0, 1.0],
        frame_scores=[0.1, 0.2],
        windows=[[0.0, 8.0], [4.0, 12.0]],
        window_scores=[0.1, 0.2],
        prediction=[4.0, 12.0],
    )

    assert set(result) == {
        "config",
        "video",
        "prediction",
        "frame_timestamps",
        "frame_scores",
        "windows",
        "window_scores",
    }
    assert result["config"] == {
        "video": "video.mp4",
        "query": "person opens a door",
        "fps": 1.0,
        "window_size": 8.0,
        "stride": 4.0,
        "aggregation": "mean",
        "model_name": "ViT-B/32",
        "device": "cpu",
    }
    assert result["video"] == {
        "duration": 12.0,
        "num_frames": 2,
    }
    assert result["prediction"] == [4.0, 12.0]


def test_one_video_clip_retrieval_script_contains_expected_pipeline_steps() -> None:
    script = Path("scripts/run_one_video_clip_retrieval.py").read_text(encoding="utf-8")

    assert "extract_sampled_frames" in script
    assert "load_clip_model" in script
    assert "encode_text" in script
    assert "encode_images" in script
    assert "compute_text_image_similarity" in script
    assert "generate_temporal_windows" in script
    assert "aggregate_frame_scores_to_windows" in script
    assert "select_best_window" in script
