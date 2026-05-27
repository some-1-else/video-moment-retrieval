import json

from scripts.run_frame_score_retrieval import parse_args, run_from_args


def test_run_frame_score_retrieval_script_writes_output_and_respects_limit(tmp_path) -> None:
    annotations_path = tmp_path / "annotations.jsonl"
    output_path = tmp_path / "results" / "frame_score.json"
    records = [
        {
            "sample_id": "sample_1",
            "video_id": "video_1",
            "query": "person opens a door",
            "duration": 8.0,
            "relevant_windows": [[2.0, 6.0]],
        },
        {
            "sample_id": "sample_2",
            "video_id": "video_2",
            "query": "person sits down",
            "duration": 12.0,
            "relevant_windows": [[4.0, 8.0]],
        },
    ]
    annotations_path.write_text(
        "\n".join(json.dumps(record) for record in records),
        encoding="utf-8",
    )
    args = parse_args(
        [
            "--annotations",
            str(annotations_path),
            "--fps",
            "1",
            "--window-size",
            "4",
            "--stride",
            "2",
            "--aggregation",
            "mean",
            "--smoothing-window",
            "3",
            "--limit",
            "1",
            "--output",
            str(output_path),
        ]
    )

    result = run_from_args(args)

    assert set(result) == {"config", "predictions", "ground_truth", "metrics", "stats"}
    assert result["config"] == {
        "fps": 1.0,
        "window_size": 4.0,
        "stride": 2.0,
        "aggregation": "mean",
        "smoothing_window": 3,
    }
    assert result["predictions"] == {"sample_1": [2.0, 6.0]}
    assert result["ground_truth"] == {"sample_1": [[2.0, 6.0]]}
    assert set(result["metrics"]) == {
        "R@1_IoU_0.3",
        "R@1_IoU_0.5",
        "R@1_IoU_0.7",
        "mIoU",
    }
    assert set(result["stats"]) == {
        "num_samples",
        "total_frames",
        "avg_frames_per_sample",
    }
    assert result["stats"]["num_samples"] == 1
    assert result["stats"]["total_frames"] == 8
    assert result["stats"]["avg_frames_per_sample"] == 8.0
    assert output_path.exists()
    assert json.loads(output_path.read_text(encoding="utf-8")) == result
