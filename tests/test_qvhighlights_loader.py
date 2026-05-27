import json

import pytest

from src.data.qvhighlights import MomentRetrievalSample, load_qvhighlights_annotations


def test_load_qvhighlights_annotations_from_json_list(tmp_path) -> None:
    path = tmp_path / "annotations.json"
    path.write_text(
        json.dumps(
            [
                {
                    "sample_id": "sample-1",
                    "video_id": "video-1",
                    "query": "person opens a door",
                    "duration": 10,
                    "relevant_windows": [[4, 8]],
                }
            ]
        ),
        encoding="utf-8",
    )

    samples = load_qvhighlights_annotations(path)

    assert samples == [
        MomentRetrievalSample(
            sample_id="sample-1",
            video_id="video-1",
            query="person opens a door",
            duration=10.0,
            relevant_windows=[[4.0, 8.0]],
        )
    ]


def test_load_qvhighlights_annotations_from_jsonl(tmp_path) -> None:
    path = tmp_path / "annotations.jsonl"
    records = [
        {
            "sample_id": "sample-1",
            "video_id": "video-1",
            "query": "person opens a door",
            "duration": 10.0,
            "relevant_windows": [[4.0, 8.0]],
        },
        {
            "sample_id": "sample-2",
            "video_id": "video-2",
            "query": "person sits down",
            "duration": 12.0,
            "relevant_windows": [[1.0, 3.0]],
        },
    ]
    path.write_text("\n".join(json.dumps(record) for record in records), encoding="utf-8")

    samples = load_qvhighlights_annotations(path)

    assert len(samples) == 2
    assert samples[0].sample_id == "sample-1"
    assert samples[1].sample_id == "sample-2"


def test_load_qvhighlights_annotations_supports_alternative_field_names(tmp_path) -> None:
    path = tmp_path / "annotations.json"
    path.write_text(
        json.dumps(
            [
                {
                    "qid": "query-1",
                    "vid": "video-1",
                    "query_txt": "person opens a door",
                    "video_duration": "10.0",
                    "timestamps": [[4, 8]],
                },
                {
                    "annotation_id": "annotation-2",
                    "video_id": "video-2",
                    "sentence": "person sits down",
                    "duration": 12,
                    "gt_windows": [[1, 3]],
                },
            ]
        ),
        encoding="utf-8",
    )

    samples = load_qvhighlights_annotations(path)

    assert samples[0] == MomentRetrievalSample(
        sample_id="query-1",
        video_id="video-1",
        query="person opens a door",
        duration=10.0,
        relevant_windows=[[4.0, 8.0]],
    )
    assert samples[1] == MomentRetrievalSample(
        sample_id="annotation-2",
        video_id="video-2",
        query="person sits down",
        duration=12.0,
        relevant_windows=[[1.0, 3.0]],
    )


def test_load_qvhighlights_annotations_raises_for_empty_relevant_windows(tmp_path) -> None:
    path = tmp_path / "annotations.json"
    path.write_text(json.dumps([_record(relevant_windows=[])]), encoding="utf-8")

    with pytest.raises(ValueError, match="relevant_windows.*must not be empty"):
        load_qvhighlights_annotations(path)


def test_load_qvhighlights_annotations_raises_for_invalid_window_format(tmp_path) -> None:
    path = tmp_path / "annotations.json"
    path.write_text(json.dumps([_record(relevant_windows=[[1.0, 2.0, 3.0]])]), encoding="utf-8")

    with pytest.raises(ValueError, match="format \\[start, end\\]"):
        load_qvhighlights_annotations(path)


def test_load_qvhighlights_annotations_raises_when_window_end_is_before_start(tmp_path) -> None:
    path = tmp_path / "annotations.json"
    path.write_text(json.dumps([_record(relevant_windows=[[4.0, 2.0]])]), encoding="utf-8")

    with pytest.raises(ValueError, match="end >= start"):
        load_qvhighlights_annotations(path)


def test_load_qvhighlights_annotations_raises_for_missing_required_field(tmp_path) -> None:
    path = tmp_path / "annotations.json"
    record = _record()
    del record["query"]
    path.write_text(json.dumps([record]), encoding="utf-8")

    with pytest.raises(ValueError, match="Missing required field 'query'"):
        load_qvhighlights_annotations(path)


def _record(**overrides):
    record = {
        "sample_id": "sample-1",
        "video_id": "video-1",
        "query": "person opens a door",
        "duration": 10.0,
        "relevant_windows": [[4.0, 8.0]],
    }
    record.update(overrides)
    return record
