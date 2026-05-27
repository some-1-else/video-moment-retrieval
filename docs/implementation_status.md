# Implementation Status

This document records the current implementation state of the Video Moment Retrieval course project.

## Implemented

- Project structure: `src/`, `scripts/`, `tests/`, `configs/`, `docs/`, `data/`, `results/`.
- Evaluation module:
  - temporal IoU;
  - maximum IoU over multiple ground-truth windows;
  - `Recall@1` at IoU 0.3, 0.5 and 0.7;
  - `mIoU`.
- QVHighlights-style annotation loader:
  - JSON and JSONL support;
  - flexible field aliases;
  - validation for duration and relevant windows.
- Real QVHighlights validation annotations were successfully inspected:
  - loader compatibility with real fields `qid`, `query`, `duration`, `vid` and `relevant_windows` was confirmed;
  - no code adaptation was required;
  - annotation-only rehearsal experiments were run on the first 100 validation samples.
- Temporal window generation for retrieval experiments.
- Frame timestamp sampling without video decoding.
- Frame-score aggregation:
  - `mean`;
  - `max`;
  - simple moving average smoothing.
- Dummy dataset retrieval:
  - artificial window scores;
  - dataset-level predictions and metrics.
- Frame-score rehearsal retrieval:
  - artificial frame scores;
  - frame timestamps;
  - window aggregation;
  - metrics and frame-count stats.
- Local one-video CLIP smoke test:
  - synthetic local video was generated for smoke testing;
  - frame extraction, CLIP text/image encoding, frame-level similarities, temporal window aggregation and best-window selection were verified on one short local video;
  - output was saved to `results/one_video_clip_retrieval_synthetic.json`.
- CLIP encoder and synthetic benchmark:
  - CLIP encoder is implemented for text/image embeddings and cosine similarity;
  - one-video CLIP smoke test was executed successfully;
  - synthetic dataset-level CLIP benchmark was executed successfully;
  - the benchmark uses real CLIP image-text similarities;
  - synthetic benchmark metrics:
    - `R@1_IoU_0.3 = 1.0000`;
    - `R@1_IoU_0.5 = 0.6667`;
    - `R@1_IoU_0.7 = 0.6667`;
    - `mIoU = 0.7778`;
  - output was saved to `results/synthetic_clip_dataset_retrieval.json`.
- First QVHighlights subset CLIP run:
  - real QVHighlights validation annotations were used;
  - 2 locally available validation clip videos were evaluated;
  - this is not a full validation benchmark and the results are preliminary;
  - metrics:
    - `R@1_IoU_0.3 = 0.0000`;
    - `R@1_IoU_0.5 = 0.0000`;
    - `R@1_IoU_0.7 = 0.0000`;
    - `mIoU = 0.0606`;
  - stats:
    - `num_samples = 2`;
    - `total_frames = 302`;
    - `avg_frames_per_sample = 151.0000`;
    - `inference_time_sec = 11.4364`;
  - output was saved to `results/qvhighlights_available_clip_retrieval_diverse.json`;
  - one attempted YouTube source failed with `ERROR: [youtube] NUsG9BgSes0: Video unavailable`;
  - a robustness fix was added to sampled frame extraction to ignore an unreadable final timestamp when video metadata slightly exceeds the actually readable tail.
- QVHighlights available-subset parameter sweep:
  - 6 configurations were tested on the 2 locally available validation clips;
  - the sweep varied temporal window size, stride and aggregation method;
  - all `Recall@1` values remained 0.0 on this tiny subset;
  - the best observed setting by `mIoU` was `window_size=16`, `stride=8`, `aggregation=mean`, with `mIoU = 0.0857`;
  - these results are preliminary and should not be interpreted as full QVHighlights benchmark performance.
- QVHighlights readable-subset sweep:
  - local video validation was used to filter out unreadable `.mp4` files before retrieval;
  - 8 configurations were tested on 9 readable QVHighlights validation clips;
  - the best setting by `mIoU` was `window_size=32`, `stride=16`, `aggregation=mean`, with `mIoU = 0.2316`;
  - the best settings by `R@1_IoU_0.3` were `window_size=16`, `stride=8`, `aggregation=mean` and `window_size=32`, `stride=16`, `aggregation=mean`, both with `R@1_IoU_0.3 = 0.3333`;
  - the results are preliminary and should not be interpreted as full QVHighlights benchmark performance.
- CLIP image embedding cache:
  - a simple `.npz` cache is implemented for per-video image embeddings;
  - the cache key depends on `video_id`, `model_name`, and `fps`;
  - cache smoke experiment on the 9 readable QVHighlights clips was completed;
  - first run: `cache_hits = 0`, `cache_misses = 9`, `inference_time_sec = 40.3804`;
  - second run: `cache_hits = 9`, `cache_misses = 0`, `inference_time_sec = 2.8471`;
  - cache size was `2591373` bytes, about 2.6 MB;
  - repeated-run speedup was approximately 14.2x;
  - tests passed with `120 passed`.
- Cached QVHighlights readable-subset sweep:
  - 8 configurations were tested with image embedding cache enabled;
  - metrics were identical to the uncached readable-subset sweep;
  - total inference time was reduced from `332.5960s` to `20.7162s`;
  - average per-run inference time was reduced from `41.5745s` to `2.5895s`;
  - total-time speedup was approximately 16.05x;
  - tests passed with `121 passed`.
- CLI scripts:
  - `scripts/inspect_annotations.py`;
  - `scripts/run_dummy_retrieval.py`;
  - `scripts/run_frame_score_retrieval.py`;
  - `scripts/collect_results.py`.
- Result collection:
  - JSON experiment outputs can be collected into a CSV summary table.
- Sample experiment workflow:
  - `scripts/run_sample_experiments.sh`;
  - sample JSON results;
  - `results/sample_experiments/summary.csv`.
- Unit tests for data loading, retrieval utilities, pipelines, scripts and sample data.

## Runnable commands

Run all tests:

```bash
.venv/bin/python -m pytest -q
```

Inspect sample annotations:

```bash
.venv/bin/python scripts/inspect_annotations.py \
  --annotations data/sample/qvhighlights_sample.jsonl \
  --limit 2
```

Run dummy window-level retrieval:

```bash
.venv/bin/python scripts/run_dummy_retrieval.py \
  --annotations data/sample/qvhighlights_sample.jsonl \
  --window-size 4 \
  --stride 2 \
  --output results/sample_dummy_retrieval.json
```

Run frame-score rehearsal retrieval:

```bash
.venv/bin/python scripts/run_frame_score_retrieval.py \
  --annotations data/sample/qvhighlights_sample.jsonl \
  --fps 1 \
  --window-size 4 \
  --stride 2 \
  --aggregation mean \
  --output results/sample_frame_score_retrieval.json
```

Run the toy sample experiment matrix:

```bash
bash scripts/run_sample_experiments.sh
```

## Not implemented yet

- Full QVHighlights video download and storage are not included.
- Dataset-wide QVHighlights video processing is not implemented.
- Full validation CLIP-based evaluation on QVHighlights has not been performed.

## Important distinction

Dummy retrieval uses artificial window scores.

Frame-score rehearsal retrieval uses artificial frame scores.

The one-video CLIP smoke test uses real CLIP similarities on a synthetic local video, but it has no ground-truth temporal annotations and is not a benchmark evaluation.

The synthetic benchmark verifies the CLIP retrieval pipeline with real CLIP image-text similarities, temporal windows, predictions and metrics. It is not a QVHighlights benchmark result, and the metrics are not representative of real dataset quality.

The first QVHighlights subset CLIP run uses real validation annotations, real downloaded clip videos and real CLIP similarities, but only on 2 locally available clips. It is a technical validation of the real-video pipeline, not a reliable benchmark estimate for QVHighlights.

The available-subset parameter sweep is also preliminary. It is useful for validating the experiment workflow and comparing configurations, but the subset is far too small for model-quality conclusions.

These are sanity-check pipelines, not final model results. Their purpose is to verify the project architecture before adding dataset-level QVHighlights video processing and CLIP-based evaluation.

## Next steps

1. Manually add real QVHighlights annotation files to `data/qvhighlights/`.
2. Run `inspect_annotations.py` on real validation annotations.
3. Run dummy and frame-score retrieval with `--limit 100`.
4. Adapt the loader if the real annotation format differs.
5. Add a real frame extraction module.
6. Add a CLIP embedding module.
7. Run the first CLIP-based baseline on a small subset.
