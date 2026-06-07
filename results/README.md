# Results

This directory contains the final saved results used in the coursework.

## Public Coursework Results

- `charades_sta_sweep_1000/` - final CLIP window/stride/aggregation sweep on a
  fixed 1,000-query Charades-STA subset.
- `charades_sta_smoothing_1000/` - final smoothing ablation on the same
  1,000-query subset.
- `clip_vs_moment_detr_50/` - CLIP results for the fixed 50-query comparison
  subset.
- `moment_detr_charades_50/` - Moment-DETR raw-video probe results on the same
  fixed 50-query subset.
- `figures/` - optional generated figures for reports or coursework materials.

The main summary files are:

- `charades_sta_sweep_1000/summary.csv`
- `charades_sta_smoothing_1000/summary.csv`
- `clip_vs_moment_detr_50/comparison_summary.csv`
- `moment_detr_charades_50/metrics.json`

## Local Research History

Exploratory outputs, old probe runs, QVHighlights preliminary results,
sample/synthetic demos, and embedding caches are preserved locally under:

```text
.agent_memory/results/
```

That directory is ignored by git and is not part of the public coursework
results surface. These files are useful for local traceability, but they are not
the evidence path for the final reported numbers.

## Caches

Embedding `.npz` files are local cache artifacts. They can speed up reruns, but
they are not standalone final results. Public result folders keep the small
metrics, configs, summaries, and prediction outputs needed to inspect the
reported experiments.
