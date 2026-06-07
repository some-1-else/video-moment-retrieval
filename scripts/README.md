# Scripts

Public reproduction entrypoints for the coursework.

## Data Preparation

- `prepare_data.py` extracts selected Charades videos from a local zip archive
  into the expected raw-video layout.

## Experiments

- `run_clip_sweep.py` runs the main CLIP window/stride/aggregation sweep on
  Charades-STA.
- `run_smoothing_sweep.py` runs the CLIP smoothing ablation on the same
  Charades-STA setup.
- `run_clip_vs_moment_detr.py` runs the CLIP side of the fixed 50-query
  comparison and writes the combined comparison table.
- `run_moment_detr_probe.py` runs the Moment-DETR raw-video feasibility probe
  on the fixed Charades-STA subset.

## Result Collection

- `collect_results.py` collects JSON result files into a compact CSV summary.

Older probe, synthetic, QVHighlights, and debug scripts were moved to local
ignored agent memory and are not part of the public reproduction path.
