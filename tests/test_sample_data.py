from pathlib import Path

from src.data.qvhighlights import load_qvhighlights_annotations


def test_sample_qvhighlights_jsonl_can_be_loaded() -> None:
    sample_path = Path("data/sample/qvhighlights_sample.jsonl")

    samples = load_qvhighlights_annotations(sample_path)

    assert len(samples) >= 3
    assert all(sample.sample_id for sample in samples)
    assert all(sample.video_id for sample in samples)
    assert all(sample.query for sample in samples)
    assert all(sample.duration > 0 for sample in samples)
    assert all(sample.relevant_windows for sample in samples)
