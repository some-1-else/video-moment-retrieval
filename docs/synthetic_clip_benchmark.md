# Synthetic CLIP Benchmark

This document describes a tiny synthetic benchmark for checking dataset-level CLIP retrieval on local videos.

## Purpose

The synthetic benchmark verifies this pipeline:

```text
synthetic annotations + synthetic local videos
-> sampled frames
-> CLIP similarities
-> temporal window scores
-> predicted moments
-> Recall@1 and mIoU
```

It uses real CLIP image-text similarity scores, but the videos and annotations are toy examples. This is not a QVHighlights benchmark and not a meaningful estimate of final retrieval quality.

## Create Synthetic Data

Generate three tiny videos and a JSONL annotation file:

```bash
.venv/bin/python scripts/create_synthetic_clip_dataset.py
```

This creates:

```text
data/sample/synthetic_clip_videos/
  synthetic_red_square.mp4
  synthetic_blue_square.mp4
  synthetic_green_square.mp4

data/sample/synthetic_clip_annotations.jsonl
```

The MP4 files are ignored by git. The annotation JSONL is small and can be kept for reproducibility.

## Run Dataset-Level CLIP Retrieval

```bash
.venv/bin/python scripts/run_clip_dataset_retrieval.py \
  --annotations data/sample/synthetic_clip_annotations.jsonl \
  --videos-dir data/sample/synthetic_clip_videos \
  --fps 1 \
  --window-size 2 \
  --stride 1 \
  --aggregation mean \
  --model-name ViT-B/32 \
  --output results/synthetic_clip_dataset_retrieval.json
```

The output JSON contains config, predictions, ground truth, metrics, and stats.

## Embedding cache

Dataset-level CLIP retrieval can optionally cache image embeddings for each video, model name, and FPS setting. This avoids re-encoding the same sampled frames when running repeated parameter sweeps over window size, stride, or aggregation.

Only image embeddings are cached. Text query embeddings are still computed for each run. Cache files are stored as `.npz` files and take additional disk space.

Example:

```bash
.venv/bin/python scripts/run_clip_dataset_retrieval.py \
  --annotations data/sample/synthetic_clip_annotations.jsonl \
  --videos-dir data/sample/synthetic_clip_videos \
  --fps 1 \
  --window-size 2 \
  --stride 1 \
  --aggregation mean \
  --model-name ViT-B/32 \
  --use-cache \
  --embeddings-cache-dir results/embeddings/synthetic \
  --output results/synthetic_clip_dataset_retrieval_cached.json
```

## Completed smoke benchmark

The following command was executed successfully:

```bash
.venv/bin/python scripts/run_clip_dataset_retrieval.py \
  --annotations data/sample/synthetic_clip_annotations.jsonl \
  --videos-dir data/sample/synthetic_clip_videos \
  --fps 1 \
  --window-size 2 \
  --stride 1 \
  --aggregation mean \
  --model-name ViT-B/32 \
  --output results/synthetic_clip_dataset_retrieval.json
```

Metrics:

```text
R@1_IoU_0.3: 1.0000
R@1_IoU_0.5: 0.6667
R@1_IoU_0.7: 0.6667
mIoU: 0.7778
```

Stats:

```text
num_samples: 3
total_frames: 15
avg_frames_per_sample: 5.0
device: cpu
```

Output:

```text
results/synthetic_clip_dataset_retrieval.json
```

These metrics confirm that the synthetic CLIP pipeline runs end-to-end. They should not be interpreted as QVHighlights results because the benchmark is tiny, synthetic and visually simple.

## Limitations

- The dataset has only three samples.
- The videos are artificial and visually simple.
- Metrics are not representative of QVHighlights performance.
- This does not train or fine-tune CLIP.
- This does not process QVHighlights videos.
