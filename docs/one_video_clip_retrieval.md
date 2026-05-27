# One-Video CLIP Retrieval

This document describes an optional smoke test for running CLIP-based moment retrieval on a single local video file.

## Purpose

The script connects the already implemented components:

```text
local video + text query
-> sampled frames
-> CLIP text and image embeddings
-> frame-level similarities
-> temporal window aggregation
-> predicted moment
```

This is the first script that can use real CLIP image-text similarities. It is still not a QVHighlights evaluation and not the final dataset-level CLIP baseline.

## Command

Run on a short local video:

```bash
.venv/bin/python scripts/run_one_video_clip_retrieval.py \
  --video path/to/video.mp4 \
  --query "person opens a door" \
  --fps 1 \
  --window-size 8 \
  --stride 4 \
  --aggregation mean \
  --model-name ViT-B/32 \
  --output results/one_video_clip_retrieval.json
```

The script prints the selected device, video duration, number of sampled frames, predicted window, and best window score.

## Output

The result JSON contains:

- configuration;
- video duration and frame count;
- predicted temporal window;
- frame timestamps;
- frame-level CLIP similarity scores;
- candidate windows;
- aggregated window scores.

## What This Is Not

- It does not process QVHighlights videos.
- It does not run dataset-level evaluation.
- It does not train or fine-tune CLIP.
- It does not cache embeddings yet.

## Common Issues

- `torch` or `clip` is not installed: install the project requirements first.
- First run downloads weights: `clip.load()` may download pretrained CLIP weights.
- Slow CPU inference: use a short video and low `--fps` for smoke tests.
- OpenCV cannot read the video: the local codec may not be supported by the installed OpenCV/FFmpeg stack.
