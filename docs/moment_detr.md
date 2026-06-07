# Moment-DETR

This document summarizes what was done with Moment-DETR in the coursework and
which limitations were found. Moment-DETR is treated as an external feasibility
comparison, not as the main reproducible pipeline.

## Role in the Project

The main coursework line is Charades-STA + CLIP. Moment-DETR was added as an
isolated raw-video probe to test whether a specialized temporal localization
model could be connected to the same local Charades videos and evaluated with
the same metrics.

No Moment-DETR model was trained, fine-tuned, or integrated into the main CLIP
pipeline.

## Official Repository

The official repository is:

```text
https://github.com/jayleicn/moment_detr
```

It was cloned under:

```text
external/moment_detr
```

Observed commit for the local probe:

```text
b7e553a
```

The official project mainly targets QVHighlights and provides two relevant
inference routes:

- dataset-level QVHighlights inference with pre-extracted features;
- raw-video prediction through the `run_on_video` demo.

The raw-video route is the one used here because the coursework data is local
Charades `.mp4` files.

## Environment Constraints

The official Moment-DETR README recommends a separate environment with Python
3.7 and PyTorch 1.9.0. The coursework `.venv` is newer, so the probe keeps
Moment-DETR isolated instead of downgrading the main project environment.

The raw-video demo also depends on packages such as `ffmpeg-python`,
`easydict`, and `tensorboard`. The isolated wrapper avoided changing the main
environment and used OpenCV for frame extraction.

## Checkpoints

Two checkpoint types were checked:

- official release QVHighlights checkpoints, including
  `ft_model_from_pt_model_e50.ckpt`;
- the raw-video-compatible checkpoint included in the official repo:
  `external/moment_detr/run_on_video/moment_detr_ckpt/model_best.ckpt`.

The raw-video-compatible checkpoint works for the local probe. The released
QVHighlights feature checkpoint is not directly compatible with the raw-video
path: it expects feature dimension `2818`, while the raw-video path produces
CLIP image features plus temporal endpoint features, i.e. `512 + 2 = 514`.

## Charades-STA Conversion

For the local probe, Charades-STA annotations were converted into a
QVHighlights-like JSONL format:

```json
{
  "qid": "charades_sta_test_0",
  "query": "person turn a light on.",
  "duration": 31.0,
  "vid": "3MSZA",
  "relevant_windows": [[24.3, 30.4]]
}
```

The converted fixed 50-query subset is stored at:

```text
data/processed/charades_sta_moment_detr_test_subset.jsonl
```

The conversion validates that local videos exist, can be opened with OpenCV,
and have readable frames.

## One-Video Raw-Video Probe

The first successful raw-video probe used:

```text
video: data/raw/charades/videos/3MSZA.mp4
query: person turn a light on.
checkpoint: external/moment_detr/run_on_video/moment_detr_ckpt/model_best.ckpt
output: .agent_memory/results/probes/moment_detr_probe/3MSZA_prediction.json
```

The top predicted window was:

```json
[26.0624, 31.9724, 0.9685]
```

This confirmed that Moment-DETR raw-video inference can produce a temporal
prediction for a local Charades video/query pair.

## 50-Query Comparison

The public comparison uses the same fixed 50 Charades-STA examples for CLIP and
Moment-DETR. Results are stored in:

```text
results/clip_vs_moment_detr_50/comparison_summary.csv
results/moment_detr_charades_50/metrics.json
```

This comparison is a feasibility probe. It is not a full official Moment-DETR
benchmark and it is not a trained Charades-STA Moment-DETR result.

## Limitations

- The official dataset-level path expects QVHighlights-style pre-extracted
  features and does not directly run on Charades-STA videos.
- The raw-video-compatible checkpoint is usable, but the released
  QVHighlights feature checkpoint is not compatible with the raw-video feature
  dimensions.
- The comparison uses only 50 examples.
- CPU inference is slow if video encoding is repeated query by query.
- A rigorous Moment-DETR benchmark would require a stable separate
  environment, careful feature setup, batching by video, and likely
  fine-tuning or dataset-specific adaptation.

Conclusion: Moment-DETR is feasible as a small isolated raw-video comparison,
but the final coursework remains centered on the reproducible Charades-STA +
CLIP baseline.
