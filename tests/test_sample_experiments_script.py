from pathlib import Path


def test_sample_experiments_script_contains_expected_commands() -> None:
    script_path = Path("scripts/run_sample_experiments.sh")
    script = script_path.read_text(encoding="utf-8")

    assert script.startswith("#!/usr/bin/env bash\nset -euo pipefail")
    assert "data/sample/qvhighlights_sample.jsonl" in script
    assert "scripts/run_frame_score_retrieval.py" in script
    assert "scripts/collect_results.py" in script
    assert 'OUTPUT_DIR="results/sample_experiments"' in script
    assert "frame_score_w4_s2_mean.json" in script
    assert "frame_score_w4_s2_max.json" in script
    assert "frame_score_w8_s4_mean.json" in script
    assert "frame_score_w8_s4_max.json" in script
    assert "frame_score_w4_s2_mean_smooth3.json" in script
    assert "summary.csv" in script
    assert "--smoothing-window 3" in script
