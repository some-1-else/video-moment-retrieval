# CLIP Baseline Plan

This document describes the planned CLIP-based baseline for Video Moment Retrieval. It is a plan and interface target, not a runnable CLIP implementation yet.

## Pipeline

The future baseline will follow this flow:

```text
video + text query
-> extract frames at fixed FPS
-> encode frames with CLIP image encoder
-> encode query with CLIP text encoder
-> compute cosine similarity per frame
-> aggregate frame scores into temporal windows
-> select best window
-> evaluate with Recall@1 IoU 0.3/0.5/0.7 and mIoU
```

The current dummy pipeline already covers the last part of this flow:

```text
temporal windows + scores -> predicted window -> metrics
```

The CLIP implementation should replace dummy scores with actual video-text similarity scores while reusing the same retrieval and evaluation utilities.

Frame sampling is currently represented by a lightweight timestamps-only utility. It can generate planned frame timestamps from `duration` and `fps` without opening video files. Real image extraction from video files will be a separate later step.

## Planned Experiments

- Frame-level retrieval: select the most relevant frame and convert it to a short temporal interval.
- Temporal windows: aggregate frame scores inside candidate windows and select the best window.
- Different `window_size` values, for example 2, 4, 8 and 16 seconds.
- Different `stride` values to study the trade-off between coverage and computation.
- Aggregation methods: `mean` and `max` over frame-level scores inside each window.
- Smoothing similarity scores with a simple moving average before window aggregation.

## Computational Metrics

In addition to retrieval quality, the experiments should report computational cost:

- inference time;
- number of processed frames;
- embedding cache size.

These metrics are important because the course project compares not only quality, but also implementation simplicity and practical cost.

## Realistic Scope

- Do not train a model from scratch.
- Use a pretrained CLIP model.
- Start with a small subset, controlled by `limit`.
- Cache frame embeddings so repeated experiments do not recompute the same video features.
- Keep the first implementation compatible with CPU, MPS or GPU through a simple `device: auto` setting.
- Keep the pipeline reproducible and easy to inspect before optimizing performance.

## Planned Modules

The implementation can be added incrementally:

```text
src/features/frame_extraction.py
src/features/clip_embeddings.py
src/video/frame_sampling.py
src/retrieval/aggregation.py
src/pipeline/clip_baseline.py
scripts/run_clip_baseline.py
```

At this stage only lightweight timestamp sampling and score aggregation utilities are implemented. Video decoding, CLIP loading and embedding extraction are intentionally left for later.
