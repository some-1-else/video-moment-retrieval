import pytest

from src.data.qvhighlights import MomentRetrievalSample
from src.pipeline.dummy_dataset_retrieval import run_dummy_dataset_retrieval


def test_run_dummy_dataset_retrieval_returns_expected_dataset_result() -> None:
    samples = [
        MomentRetrievalSample(
            sample_id="sample_1",
            video_id="video_1",
            query="person opens a door",
            duration=8.0,
            relevant_windows=[[2.0, 6.0]],
        ),
        MomentRetrievalSample(
            sample_id="sample_2",
            video_id="video_2",
            query="person sits down",
            duration=12.0,
            relevant_windows=[[4.0, 8.0]],
        ),
    ]

    result = run_dummy_dataset_retrieval(samples, window_size=4.0, stride=2.0)

    assert isinstance(result, dict)
    assert result["config"] == {
        "window_size": 4.0,
        "stride": 2.0,
        "num_samples": 2,
    }
    assert result["predictions"] == {
        "sample_1": [2.0, 6.0],
        "sample_2": [4.0, 8.0],
    }
    assert result["ground_truth"] == {
        "sample_1": [[2.0, 6.0]],
        "sample_2": [[4.0, 8.0]],
    }
    assert set(result["metrics"]) == {
        "R@1_IoU_0.3",
        "R@1_IoU_0.5",
        "R@1_IoU_0.7",
        "mIoU",
    }
    assert result["metrics"]["R@1_IoU_0.3"] == pytest.approx(1.0)
    assert result["metrics"]["R@1_IoU_0.5"] == pytest.approx(1.0)
    assert result["metrics"]["R@1_IoU_0.7"] == pytest.approx(1.0)
    assert result["metrics"]["mIoU"] == pytest.approx(1.0)


def test_run_dummy_dataset_retrieval_raises_for_empty_samples() -> None:
    with pytest.raises(ValueError, match="samples must not be empty"):
        run_dummy_dataset_retrieval([], window_size=4.0, stride=2.0)


def test_run_dummy_dataset_retrieval_raises_for_non_positive_window_size() -> None:
    with pytest.raises(ValueError, match="window_size must be a positive number"):
        run_dummy_dataset_retrieval(_samples(), window_size=0.0, stride=2.0)


def test_run_dummy_dataset_retrieval_raises_for_non_positive_stride() -> None:
    with pytest.raises(ValueError, match="stride must be a positive number"):
        run_dummy_dataset_retrieval(_samples(), window_size=4.0, stride=0.0)


def _samples() -> list[MomentRetrievalSample]:
    return [
        MomentRetrievalSample(
            sample_id="sample_1",
            video_id="video_1",
            query="person opens a door",
            duration=8.0,
            relevant_windows=[[2.0, 6.0]],
        )
    ]
