from pathlib import Path


def test_qvhighlights_annotation_experiments_script_contains_expected_commands() -> None:
    script_path = Path("scripts/run_qvhighlights_annotation_experiments.sh")
    script = script_path.read_text(encoding="utf-8")

    assert script.startswith("#!/usr/bin/env bash\nset -euo pipefail")
    assert "data/qvhighlights/highlight_val_release.jsonl" in script
    assert "Missing annotation file:" in script
    assert "docs/qvhighlights_data.md" in script
    assert "scripts/inspect_annotations.py" in script
    assert "scripts/run_frame_score_retrieval.py" in script
    assert "scripts/collect_results.py" in script
    assert "--limit 100" in script
    assert "results/qvhighlights_annotation_rehearsal" in script
    assert "frame_score_limit100_w4_s2_mean.json" in script
    assert "frame_score_limit100_w4_s2_max.json" in script
    assert "frame_score_limit100_w8_s4_mean.json" in script
    assert "frame_score_limit100_w8_s4_max.json" in script
    assert "frame_score_limit100_w8_s4_mean_smooth3.json" in script
    assert "summary.csv" in script
