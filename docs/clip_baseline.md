# CLIP Baseline

The main coursework baseline uses pretrained CLIP as a frozen image-text
encoder for raw-video moment retrieval on Charades-STA. No CLIP parameters are
trained or fine-tuned.

## Pipeline

The baseline follows this flow:

```text
video + text query
-> sample frames at fixed FPS
-> encode frames with CLIP image encoder
-> encode query with CLIP text encoder
-> compute frame-level cosine similarity
-> aggregate scores over temporal windows
-> select the best window
-> evaluate with R@1 IoU 0.3/0.5/0.7 and mIoU
```

The public implementation keeps the core logic in `src/` and exposes final
experiment entry points through `scripts/run_clip_sweep.py` and
`scripts/run_smoothing_sweep.py`.

## CLIP Encoder

The encoder layer is responsible for:

- loading a pretrained CLIP model;
- encoding a text query into a normalized text embedding;
- encoding decoded video frames into normalized image embeddings;
- computing cosine similarity between text and image embeddings.

The intended dependencies are:

```text
torch
torchvision
ftfy
regex
tqdm
git+https://github.com/openai/CLIP.git
```

`load_clip_model(..., device="auto")` selects `cuda`, then `mps`, then `cpu`
depending on local availability. The reported coursework runs use CPU.

## Frame Extraction

Frames are sampled from local video files at fixed timestamps. The frame
extraction details are documented separately in `docs/video_frame_extraction.md`.
The final experiments use `fps=1`.

## Temporal Windows

Frame scores are aggregated into candidate temporal windows. The main CLIP sweep
uses controlled combinations of:

- window size;
- stride;
- aggregation method.

The final 1,000-query Charades-STA sweep evaluates 8-second, 16-second, and
32-second windows with mean or max aggregation.

## Scoring and Aggregation

For each query-video pair:

1. CLIP encodes the query once.
2. CLIP image embeddings are produced or loaded from cache for sampled frames.
3. Cosine similarity gives a frame-level relevance score.
4. Scores inside each candidate temporal window are aggregated with `mean` or
   `max`.
5. The highest-scoring window is used as the top-1 prediction.

The coursework results show that mean aggregation is more stable than max
aggregation in the tested Charades-STA settings.

## Smoothing

Optional smoothing applies a moving average to frame-level similarity scores
before window aggregation. The smoothing sweep compares no smoothing,
`moving_average_3`, and `moving_average_5` for selected base CLIP
configurations.

Smoothing slightly improves strict localization for the shorter `w8/s4/mean`
configuration, but it does not consistently improve all metrics.

## Embedding Cache

Image embedding extraction is the expensive repeated step. The public CLIP
scripts can reuse cached image embeddings across parameter sweeps so that
window size, stride, aggregation, and smoothing can change without recomputing
the same video features.

Embeddings and cache files are local artifacts and are not part of the final
public result surface.

## Scope

This baseline is intentionally simple:

- no model training;
- no fine-tuning on Charades-STA;
- no full QVHighlights benchmark;
- no learned temporal dynamics beyond fixed windows, aggregation, and optional
  smoothing.
