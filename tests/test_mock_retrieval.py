import pytest

from src.pipeline.mock_retrieval import run_mock_retrieval_example


def test_run_mock_retrieval_example_returns_expected_result() -> None:
    result = run_mock_retrieval_example()

    assert isinstance(result, dict)
    assert result["prediction"] == [4.0, 8.0]
    assert result["iou"] == pytest.approx(1.0)
    assert result["metrics"]["R@1_IoU_0.3"] == pytest.approx(1.0)
    assert result["metrics"]["R@1_IoU_0.5"] == pytest.approx(1.0)
    assert result["metrics"]["R@1_IoU_0.7"] == pytest.approx(1.0)
    assert result["metrics"]["mIoU"] == pytest.approx(1.0)
