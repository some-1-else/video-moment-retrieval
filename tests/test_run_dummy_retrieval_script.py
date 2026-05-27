import json

import pytest

from scripts.run_dummy_retrieval import parse_args, run_dummy_retrieval_cli


def test_run_dummy_retrieval_cli_writes_output_and_respects_limit(tmp_path) -> None:
    annotations_path = tmp_path / "annotations.jsonl"
    output_path = tmp_path / "results" / "dummy.json"
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
            "--window-size",
            "4",
            "--stride",
            "2",
            "--limit",
            "1",
            "--output",
            str(output_path),
        ]
    )

    result = run_dummy_retrieval_cli(args)

    assert set(result) == {"config", "predictions", "ground_truth", "metrics"}
    assert result["config"] == {
        "window_size": 4.0,
        "stride": 2.0,
        "num_samples": 1,
    }
    assert result["predictions"] == {"sample_1": [2.0, 6.0]}
    assert result["ground_truth"] == {"sample_1": [[2.0, 6.0]]}
    assert set(result["metrics"]) == {
        "R@1_IoU_0.3",
        "R@1_IoU_0.5",
        "R@1_IoU_0.7",
        "mIoU",
    }
    assert output_path.exists()
    saved_result = json.loads(output_path.read_text(encoding="utf-8"))
    assert saved_result == result


def test_run_dummy_retrieval_cli_raises_for_non_positive_limit(tmp_path) -> None:
    annotations_path = tmp_path / "annotations.json"
    annotations_path.write_text(
        json.dumps(
            [
                {
                    "sample_id": "sample_1",
                    "video_id": "video_1",
                    "query": "person opens a door",
                    "duration": 8.0,
                    "relevant_windows": [[2.0, 6.0]],
                }
            ]
        ),
        encoding="utf-8",
    )
    args = parse_args(
        [
            "--annotations",
            str(annotations_path),
            "--window-size",
            "4",
            "--stride",
            "2",
            "--limit",
            "0",
        ]
    )

    with pytest.raises(ValueError, match="limit must be a positive integer"):
        run_dummy_retrieval_cli(args)
