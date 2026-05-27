import json

import pytest

from scripts.inspect_annotations import parse_args, run_from_args, summarize_annotations
from src.data.qvhighlights import MomentRetrievalSample


def test_summarize_annotations_returns_expected_stats() -> None:
    samples = [
        MomentRetrievalSample(
            sample_id="sample_1",
            video_id="video_1",
            query="person opens a door",
            duration=10.0,
            relevant_windows=[[4.0, 8.0]],
        ),
        MomentRetrievalSample(
            sample_id="sample_2",
            video_id="video_2",
            query="person sits down",
            duration=12.0,
            relevant_windows=[[3.0, 5.0], [7.0, 9.0]],
        ),
    ]

    summary = summarize_annotations(samples, preview_limit=1)

    assert summary["num_samples"] == 2
    assert summary["preview"] == [
        {
            "sample_id": "sample_1",
            "video_id": "video_1",
            "query": "person opens a door",
            "duration": 10.0,
            "relevant_windows": [[4.0, 8.0]],
        }
    ]
    assert summary["min_duration"] == pytest.approx(10.0)
    assert summary["max_duration"] == pytest.approx(12.0)
    assert summary["avg_relevant_windows_per_sample"] == pytest.approx(1.5)


def test_inspect_annotations_run_from_args_loads_jsonl_and_respects_limit(tmp_path) -> None:
    annotations_path = tmp_path / "annotations.jsonl"
    records = [
        {
            "sample_id": "sample_1",
            "video_id": "video_1",
            "query": "person opens a door",
            "duration": 10.0,
            "relevant_windows": [[4.0, 8.0]],
        },
        {
            "sample_id": "sample_2",
            "video_id": "video_2",
            "query": "person sits down",
            "duration": 12.0,
            "relevant_windows": [[3.0, 5.0]],
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
            "--limit",
            "1",
        ]
    )

    summary = run_from_args(args)

    assert summary["num_samples"] == 1
    assert summary["preview"][0]["sample_id"] == "sample_1"
    assert summary["min_duration"] == pytest.approx(10.0)
    assert summary["max_duration"] == pytest.approx(10.0)


def test_inspect_annotations_run_from_args_raises_for_non_positive_limit(tmp_path) -> None:
    annotations_path = tmp_path / "annotations.json"
    annotations_path.write_text(
        json.dumps(
            [
                {
                    "sample_id": "sample_1",
                    "video_id": "video_1",
                    "query": "person opens a door",
                    "duration": 10.0,
                    "relevant_windows": [[4.0, 8.0]],
                }
            ]
        ),
        encoding="utf-8",
    )
    args = parse_args(
        [
            "--annotations",
            str(annotations_path),
            "--limit",
            "0",
        ]
    )

    with pytest.raises(ValueError, match="limit must be a positive integer"):
        run_from_args(args)
