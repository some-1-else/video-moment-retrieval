import pytest

from src.data.qvhighlights import MomentRetrievalSample
from src.pipeline.frame_score_retrieval import run_frame_score_retrieval


def test_run_frame_score_retrieval_with_mean_aggregation() -> None:
    samples = _samples()

    result = run_frame_score_retrieval(
        samples=samples,
        fps=1.0,
        window_size=4.0,
        stride=2.0,
        aggregation="mean",
    )

    assert isinstance(result, dict)
    assert set(result) == {"config", "predictions", "ground_truth", "metrics", "stats"}
    assert result["config"] == {
        "fps": 1.0,
        "window_size": 4.0,
        "stride": 2.0,
        "aggregation": "mean",
        "smoothing_window": None,
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
    assert result["stats"]["num_samples"] == 2
    assert result["stats"]["total_frames"] == 20
    assert result["stats"]["avg_frames_per_sample"] == pytest.approx(10.0)


def test_run_frame_score_retrieval_with_max_aggregation() -> None:
    result = run_frame_score_retrieval(
        samples=_samples(),
        fps=1.0,
        window_size=4.0,
        stride=2.0,
        aggregation="max",
    )

    assert result["config"]["aggregation"] == "max"
    assert len(result["predictions"]) == 2


def test_run_frame_score_retrieval_with_smoothing_window() -> None:
    result = run_frame_score_retrieval(
        samples=_samples(),
        fps=1.0,
        window_size=4.0,
        stride=2.0,
        aggregation="mean",
        smoothing_window=3,
    )

    assert result["config"]["smoothing_window"] == 3
    assert len(result["predictions"]) == 2


def test_run_frame_score_retrieval_raises_for_empty_samples() -> None:
    with pytest.raises(ValueError, match="samples must not be empty"):
        run_frame_score_retrieval([], fps=1.0, window_size=4.0, stride=2.0)


def test_run_frame_score_retrieval_raises_for_non_positive_fps() -> None:
    with pytest.raises(ValueError, match="fps must be a positive number"):
        run_frame_score_retrieval(_samples(), fps=0.0, window_size=4.0, stride=2.0)


def test_run_frame_score_retrieval_raises_for_non_positive_window_size() -> None:
    with pytest.raises(ValueError, match="window_size must be a positive number"):
        run_frame_score_retrieval(_samples(), fps=1.0, window_size=0.0, stride=2.0)


def test_run_frame_score_retrieval_raises_for_non_positive_stride() -> None:
    with pytest.raises(ValueError, match="stride must be a positive number"):
        run_frame_score_retrieval(_samples(), fps=1.0, window_size=4.0, stride=0.0)


def test_run_frame_score_retrieval_raises_for_unknown_aggregation() -> None:
    with pytest.raises(ValueError, match="method must be one of"):
        run_frame_score_retrieval(
            _samples(),
            fps=1.0,
            window_size=4.0,
            stride=2.0,
            aggregation="median",
        )


def _samples() -> list[MomentRetrievalSample]:
    return [
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
