import csv
import json

from scripts.collect_results import CSV_COLUMNS, collect_result_rows, write_summary_csv


def test_collect_result_rows_reads_multiple_json_files_and_flattens_fields(tmp_path) -> None:
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    _write_json(
        results_dir / "dummy.json",
        {
            "config": {
                "window_size": 4.0,
                "stride": 2.0,
                "num_samples": 2,
            },
            "metrics": {
                "R@1_IoU_0.3": 1.0,
                "R@1_IoU_0.5": 0.5,
                "R@1_IoU_0.7": 0.0,
                "mIoU": 0.6,
            },
        },
    )
    _write_json(
        results_dir / "frame_score.json",
        {
            "config": {
                "fps": 1.0,
                "window_size": 4.0,
                "stride": 2.0,
                "aggregation": "mean",
                "smoothing_window": 3,
            },
            "metrics": {
                "R@1_IoU_0.3": 1.0,
                "R@1_IoU_0.5": 1.0,
                "R@1_IoU_0.7": 0.5,
                "mIoU": 0.75,
            },
            "stats": {
                "num_samples": 2,
                "total_frames": 20,
                "avg_frames_per_sample": 10.0,
                "inference_time_sec": 1.25,
            },
        },
    )

    rows = collect_result_rows(results_dir)

    assert [row["result_file"] for row in rows] == ["dummy.json", "frame_score.json"]
    assert rows[0]["num_samples"] == 2
    assert rows[0]["fps"] == ""
    assert rows[0]["total_frames"] == ""
    assert rows[0]["R@1_IoU_0.5"] == 0.5
    assert rows[1]["fps"] == 1.0
    assert rows[1]["aggregation"] == "mean"
    assert rows[1]["smoothing_window"] == 3
    assert rows[1]["total_frames"] == 20
    assert rows[1]["inference_time_sec"] == 1.25
    assert rows[1]["cache_hits"] == ""
    assert rows[1]["cache_misses"] == ""
    assert rows[1]["embedding_cache_size_bytes"] == ""


def test_collect_result_rows_handles_json_without_metrics_with_empty_metric_fields(tmp_path) -> None:
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    _write_json(results_dir / "metadata.json", {"config": {"window_size": 4.0}})

    rows = collect_result_rows(results_dir)

    assert rows == [
        {
            "result_file": "metadata.json",
            "num_samples": "",
            "fps": "",
            "window_size": 4.0,
            "stride": "",
            "aggregation": "",
            "smoothing_window": "",
            "R@1_IoU_0.3": "",
            "R@1_IoU_0.5": "",
            "R@1_IoU_0.7": "",
            "mIoU": "",
            "total_frames": "",
            "avg_frames_per_sample": "",
            "inference_time_sec": "",
            "cache_hits": "",
            "cache_misses": "",
            "embedding_cache_size_bytes": "",
        }
    ]


def test_write_summary_csv_creates_csv_with_expected_columns(tmp_path) -> None:
    output_path = tmp_path / "nested" / "summary.csv"
    rows = [
        {
            "result_file": "frame_score.json",
            "num_samples": 2,
            "fps": 1.0,
            "window_size": 4.0,
            "stride": 2.0,
            "aggregation": "mean",
            "smoothing_window": 3,
            "R@1_IoU_0.3": 1.0,
            "R@1_IoU_0.5": 1.0,
            "R@1_IoU_0.7": 0.5,
            "mIoU": 0.75,
            "total_frames": 20,
            "avg_frames_per_sample": 10.0,
            "inference_time_sec": 1.25,
            "cache_hits": 2,
            "cache_misses": 0,
            "embedding_cache_size_bytes": 1024,
        }
    ]

    write_summary_csv(rows, output_path)

    assert output_path.exists()
    with output_path.open("r", newline="", encoding="utf-8") as output_file:
        reader = csv.DictReader(output_file)
        assert reader.fieldnames == CSV_COLUMNS
        csv_rows = list(reader)

    assert csv_rows[0]["result_file"] == "frame_score.json"
    assert csv_rows[0]["R@1_IoU_0.3"] == "1.0"
    assert csv_rows[0]["total_frames"] == "20"
    assert csv_rows[0]["inference_time_sec"] == "1.25"
    assert csv_rows[0]["cache_hits"] == "2"
    assert csv_rows[0]["cache_misses"] == "0"
    assert csv_rows[0]["embedding_cache_size_bytes"] == "1024"


def _write_json(path, value) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")
