# Project Status

## What Is Implemented

- Charades-STA loader and local dataset probe.
- CLIP-based video moment retrieval baseline with frame sampling, CLIP
  image/text encoding, temporal windowing, mean/max aggregation, optional
  smoothing, and embedding cache.
- Evaluation with `R@1` at IoU `0.3`, `0.5`, `0.7` and `mIoU`.
- Controlled Charades-STA sweeps over window size, stride, aggregation, and
  smoothing.
- Isolated Moment-DETR raw-video probe using the external official repository
  under `external/moment_detr`.
- Lightweight notebooks and result tables for coursework review.

## What Is Tested

The repository keeps tests for:

- data loading and annotation parsing;
- temporal IoU and metric calculation;
- window generation and score aggregation;
- frame sampling and frame extraction;
- CLIP encoder/cache interfaces;
- script-level smoke tests for dataset probes and retrieval runners.

The tests remain in `tests/` and were not removed. They are useful for
development confidence, but they are not the main interface for reading the
coursework results.

## Main Results

### CLIP Sweep on 1,000 Charades-STA Queries

- Best coarse result: `clip_w16_s8_mean`, `R@1_IoU_0.3 = 0.625`.
- Best strict localization: `clip_w8_s4_mean`,
  `R@1_IoU_0.5 = 0.393`, `R@1_IoU_0.7 = 0.181`.
- Best mIoU in the sweep: `clip_w16_s8_mean`, `mIoU = 0.34969020291901676`.

Source: `results/charades_sta_sweep_1000/summary.csv`.

### Smoothing

For `w8/s4/mean`, moving average smoothing slightly improves strict metrics:

- no smoothing: `R@1_IoU_0.7 = 0.181`, `mIoU = 0.34777747887511207`;
- moving average 5: `R@1_IoU_0.7 = 0.190`,
  `mIoU = 0.34942105981287236`.

For `w16/s8/mean`, smoothing does not improve mIoU over the no-smoothing run.

Source: `results/charades_sta_smoothing_1000/summary.csv`.

### CLIP vs Moment-DETR on 50 Queries

On the fixed 50-query comparison subset:

- best CLIP strict result: `clip_w8_s4_mean`,
  `R@1_IoU_0.7 = 0.32`, `mIoU = 0.4393782459360136`;
- Moment-DETR raw-video probe:
  `R@1_IoU_0.3 = 0.44`, `R@1_IoU_0.5 = 0.32`,
  `R@1_IoU_0.7 = 0.04`, `mIoU = 0.26340237468832634`.

Sources:

- `results/clip_vs_moment_detr_50/*/run_config.json`
- `results/moment_detr_charades_50/metrics.json`

## Limitations

- Main CLIP sweep uses a fixed 1,000-query subset, not the full Charades-STA
  test split.
- Moment-DETR comparison uses only 50 examples and should be read as a
  feasibility probe, not a full official benchmark.
- All reported runs use CPU inference.
- No model is fine-tuned on Charades-STA.
- QVHighlights remains preliminary because raw video availability depends on
  external YouTube links.
- Moment-DETR checkpoint compatibility is sensitive to feature format: the
  raw-video-compatible checkpoint works for the probe, while the QVHighlights
  feature checkpoint is not directly compatible with the raw-video path.

## Next Steps

- Run the CLIP baseline on a larger or full Charades-STA test split if time and
  compute allow.
- Add GPU timing or separate CPU/GPU runtime reporting.
- Make the Moment-DETR comparison larger only after stabilizing batching and
  checkpoint/feature compatibility.
- Add qualitative examples with predicted and ground-truth windows.
- Polish the final coursework draft using the result tables in `reports/`.
