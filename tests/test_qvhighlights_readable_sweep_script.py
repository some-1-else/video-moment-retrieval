from pathlib import Path


def test_qvhighlights_readable_sweep_script_contains_expected_commands() -> None:
    script_path = Path("scripts/run_qvhighlights_readable_sweep.sh")

    assert script_path.exists()
    script_text = script_path.read_text(encoding="utf-8")

    assert "run_qvhighlights_available_clip_retrieval.py" in script_text
    assert "--validation-report" in script_text
    assert "collect_results.py" in script_text
    assert "data/qvhighlights/qvhighlights_val_diverse50_subset_manifest.json" in script_text
    assert "data/qvhighlights/qvhighlights_val_diverse50_video_validation.json" in script_text
    assert "results/qvhighlights_readable_sweep" in script_text

    expected_outputs = [
        "clip_w4_s2_mean.json",
        "clip_w4_s2_max.json",
        "clip_w8_s4_mean.json",
        "clip_w8_s4_max.json",
        "clip_w16_s8_mean.json",
        "clip_w16_s8_max.json",
        "clip_w32_s16_mean.json",
        "clip_w32_s16_max.json",
    ]
    for output_name in expected_outputs:
        assert output_name in script_text

    for window_size in ("4", "8", "16", "32"):
        assert f"--window-size {window_size}" in script_text

    for stride in ("2", "4", "8", "16"):
        assert f"--stride {stride}" in script_text

    assert "--aggregation mean" in script_text
    assert "--aggregation max" in script_text
