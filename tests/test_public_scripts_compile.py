import py_compile
from pathlib import Path


PUBLIC_SCRIPTS = [
    Path("scripts/prepare_data.py"),
    Path("scripts/run_clip_sweep.py"),
    Path("scripts/run_smoothing_sweep.py"),
    Path("scripts/run_clip_vs_moment_detr.py"),
    Path("scripts/run_moment_detr_probe.py"),
    Path("scripts/collect_results.py"),
    Path("scripts/build_clip_vs_moment_detr_comparison.py"),
]


def test_public_scripts_compile() -> None:
    for script_path in PUBLIC_SCRIPTS:
        assert script_path.exists()
        py_compile.compile(str(script_path), doraise=True)
