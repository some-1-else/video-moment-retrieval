# Video Moment Retrieval Using Text Queries

## 1. Introduction

Video Moment Retrieval is the task of localizing a temporal segment in a video
that corresponds to a natural-language query. Given a video and a sentence such
as "person turn a light on", the system must predict the start and end time of
the relevant moment. This task is important for video search, video question
answering, and human-centered media navigation, because users usually describe
events in language rather than by frame indices.

The original plan for this coursework was to use QVHighlights as the main
dataset. QVHighlights is directly connected to modern moment retrieval research
and is used by Moment-DETR. However, its raw videos are based on YouTube clips,
which created practical availability problems: some videos are unavailable,
private, deleted, or difficult to reproduce locally. For a coursework project,
this made the full raw-video pipeline less reliable.

For this reason, the main experimental dataset was changed to Charades-STA.
Charades-STA is a standard temporal sentence grounding dataset built on the
Charades videos. In this project, Charades-STA is used together with locally
extracted Charades 480p videos. This gives a reproducible raw-video setup while
keeping the task aligned with Video Moment Retrieval.

The project implements and evaluates a CLIP-based retrieval baseline. The
baseline samples video frames, computes image-text similarities with CLIP, turns
frame scores into temporal window scores, and selects the top temporal window.
The implementation also includes temporal aggregation, optional similarity
smoothing, and an embedding cache for efficient sweeps. In addition, the project
includes a limited Moment-DETR raw-video feasibility probe adapted from the
official external repository. The Moment-DETR comparison is intentionally
limited and should not be interpreted as a full official benchmark.

## 2. Related Work

Temporal sentence grounding and Video Moment Retrieval study how to match a
language query to a temporal segment in an untrimmed video. Earlier and standard
datasets include Charades-STA, which provides sentence annotations with start
and end timestamps over Charades videos. QVHighlights extends the task to a
highlight-oriented setting and is used by Moment-DETR.

CLIP is a vision-language model trained on image-text pairs. Although CLIP is
not a temporal localization model, it provides a strong zero-shot image-text
similarity signal. This makes it useful as a simple baseline for frame-level
video retrieval: sampled frames can be compared with a query, and the resulting
similarities can be aggregated over temporal windows.

Moment-DETR is a transformer-based moment retrieval model released with
QVHighlights. It is more specialized than a simple CLIP baseline because it
models temporal proposals and relevance jointly. However, the official
dataset-level inference path expects pre-extracted QVHighlights-style features,
while the raw-video demo path uses a different checkpoint and feature format.
In this coursework, Moment-DETR is therefore used only as an isolated raw-video
probe, not as a fully reproduced official benchmark.

## 3. Datasets

### 3.1 QVHighlights Initial Plan

QVHighlights was the initial target dataset because it is closely related to
Moment-DETR and modern Video Moment Retrieval evaluation. The project keeps the
QVHighlights loader code, preliminary checks, and early local experiments.
However, QVHighlights raw videos depend on YouTube availability, which made the
dataset difficult to reproduce reliably within the coursework timeline.

The QVHighlights code and preliminary results were not removed. They remain as
part of the project history and can be reused if a stable raw-video subset
becomes available. The final experimental results in this draft, however, are
based on Charades-STA.

### 3.2 Charades-STA Final Experimental Dataset

Charades-STA was selected as the final experimental dataset because it provides
temporal sentence grounding annotations over Charades videos, and the raw video
archive can be prepared locally. The local setup used in this project contains:

- Charades 480p videos extracted locally: 9,848 `.mp4` files under
  `data/raw/charades/videos/`.
- Charades-STA train annotations: 12,408 rows.
- Charades-STA test annotations: 3,720 rows.
- Annotation files under `data/charades_sta/`.

The Charades-STA annotation format is line-based:

```text
video_id start_time end_time##sentence
```

The implemented Charades-STA loader converts these rows into the internal
moment retrieval format:

```json
{
  "qid": "charades_sta_test_0",
  "vid": "3MSZA",
  "query": "person turn a light on.",
  "duration": 31.0,
  "relevant_windows": [[24.3, 30.4]],
  "source_dataset": "charades_sta"
}
```

Video duration is read from local video metadata with OpenCV when it is not
available in annotations. Ground-truth windows that exceed the decoded video
duration are clipped to the valid interval `[0, duration]` in the experiment
runners, and the number of clipped windows is logged.

## 4. Methodology

### 4.1 Implemented CLIP-Based Retrieval

The main method implemented by us is a CLIP-based frame and temporal-window
retrieval baseline. It does not train or fine-tune a model. It uses pretrained
CLIP ViT-B/32 to compute visual and textual embeddings.

For each video-query pair, the pipeline:

1. Samples video frames at a fixed rate, using `fps=1` in the reported
   experiments.
2. Encodes sampled frames with the CLIP image encoder.
3. Encodes the query with the CLIP text encoder.
4. Computes frame-level image-text similarity scores.
5. Optionally smooths the similarity sequence.
6. Generates temporal candidate windows.
7. Aggregates frame scores inside each candidate window.
8. Selects the highest-scoring window.
9. Evaluates the prediction against the ground-truth temporal window.

The retrieval and evaluation pipeline is implemented inside the project code.
The saved experiment results are under `results/`.

### 4.2 Temporal Windows

The baseline converts frame-level similarities into temporal moment predictions
by evaluating fixed-size candidate windows. Each configuration is defined by a
window size and stride in seconds. For example, `w16/s8` means a 16-second
window with an 8-second stride.

This design provides a controlled way to study the trade-off between coarse and
strict localization. Shorter windows can localize actions more precisely, while
longer windows may capture broader context and improve coarse overlap at lower
IoU thresholds.

### 4.3 Aggregation

Two aggregation strategies are evaluated:

- Mean aggregation: average the CLIP frame similarities inside a window.
- Max aggregation: use the highest frame similarity inside a window.

Mean aggregation is expected to be more stable when several frames in a moment
are moderately relevant. Max aggregation may help if a single frame is highly
diagnostic, but it can also be more sensitive to noisy frame-level matches.

### 4.4 Similarity Smoothing

The smoothing experiment applies a moving average to frame-level similarity
scores before temporal window aggregation. The tested variants are:

- no smoothing;
- moving average with window size 3;
- moving average with window size 5.

Smoothing is implemented as an isolated extension to the CLIP retrieval runner.
It does not change the global evaluation logic. The goal is to check whether
reducing frame-score noise improves temporal localization on the same fixed
1,000-query Charades-STA subset.

### 4.5 Embedding Cache

The CLIP image encoder is the most expensive part of the baseline. The project
therefore uses a per-video embedding cache. The cache key depends on the video
identifier, CLIP model name, and sampling FPS. Text embeddings are computed per
query, while video image embeddings can be reused across different window sizes,
strides, aggregation methods, and smoothing variants.

The cache is especially important for controlled sweeps. In the 1,000-query
Charades-STA sweep, the first configuration populated the cache and took
309.6866 seconds. Later cached configurations took approximately 21-23 seconds
in the main sweep, and the smoothing runs reused the same cache with 1,000 cache
hits and 0 misses.

### 4.6 Moment-DETR Raw-Video Probe

Moment-DETR was not implemented from scratch. The project cloned the official
external repository into `external/moment_detr` and used its raw-video-compatible
checkpoint:

```text
external/moment_detr/run_on_video/moment_detr_ckpt/model_best.ckpt
```

The isolated wrapper added in this project adapts the official raw-video model
path to local Charades videos and evaluates top-1 predicted windows with the
same local metrics as the CLIP baseline. This wrapper is not part of the main
CLIP pipeline.

The official QVHighlights release checkpoint was also checked, but it was not
compatible with the raw-video path. The release checkpoint expects feature
dimension 2818, while the raw-video path produces CLIP image features plus
temporal endpoint features, i.e. `512 + 2 = 514`. This finding is documented in
`docs/moment_detr.md`.

## 5. Experimental Setup

### 5.1 Hardware and Device

All reported experiments in this draft use CPU inference. No GPU acceleration
is assumed. This makes the runtime numbers useful for local reproducibility,
but they should not be interpreted as optimized inference throughput.

### 5.2 Charades-STA 1,000-Query Fixed Subset

The main CLIP experiments use a fixed 1,000-query subset from the Charades-STA
test split. The subset contains 363 unique videos. All CLIP sweep and smoothing
configurations use exactly the same selection manifest, so the reported
differences come from retrieval configuration rather than different data
selection.

The main 1,000-query sweep is stored in:

```text
results/charades_sta_sweep_1000/summary.csv
```

The smoothing experiment is stored in:

```text
results/charades_sta_smoothing_1000/summary.csv
```

The 50-query CLIP vs Moment-DETR comparison is stored in:

```text
results/clip_vs_moment_detr_50/comparison_summary.csv
results/moment_detr_charades_50/metrics.json
```

### 5.3 Metrics

The evaluation uses standard temporal localization metrics:

- `R@1 IoU 0.3`: fraction of queries where the top predicted window overlaps
  the ground-truth window with IoU at least 0.3.
- `R@1 IoU 0.5`: same at IoU threshold 0.5.
- `R@1 IoU 0.7`: same at IoU threshold 0.7.
- `mIoU`: mean temporal IoU between the top prediction and the ground-truth
  window.

All methods are evaluated with top-1 temporal predictions.

### 5.4 CLIP Configurations

The main 1,000-query CLIP sweep evaluates:

- `window_size=8`, `stride=4`, `aggregation=mean`;
- `window_size=16`, `stride=8`, `aggregation=mean`;
- `window_size=32`, `stride=16`, `aggregation=mean`;
- `window_size=16`, `stride=8`, `aggregation=max`.

All configurations use:

- `fps=1`;
- CLIP `ViT-B/32`;
- CPU;
- shared embedding cache.

The smoothing experiment evaluates two base configurations:

- `w8/s4/mean`;
- `w16/s8/mean`;

with smoothing variants `none`, `moving_average_3`, and `moving_average_5`.

## 6. Results

### 6.1 CLIP 1,000-Query Sweep

The following table reports the main controlled sweep on the fixed 1,000-query
Charades-STA test subset. The source file is
`results/charades_sta_sweep_1000/summary.csv`.

| Configuration | Window | Stride | Aggregation | Queries | Unique Videos | Frames | R@1 IoU 0.3 | R@1 IoU 0.5 | R@1 IoU 0.7 | mIoU | Time (s) | Cache Hits | Cache Misses |
|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| CLIP w8/s4/mean | 8 | 4 | mean | 1000 | 363 | 29994 | 0.509 | 0.393 | 0.181 | 0.3478 | 309.69 | 637 | 363 |
| CLIP w16/s8/mean | 16 | 8 | mean | 1000 | 363 | 29994 | 0.625 | 0.260 | 0.096 | 0.3497 | 22.59 | 1000 | 0 |
| CLIP w32/s16/mean | 32 | 16 | mean | 1000 | 363 | 29994 | 0.395 | 0.014 | 0.000 | 0.2814 | 21.51 | 1000 | 0 |
| CLIP w16/s8/max | 16 | 8 | max | 1000 | 363 | 29994 | 0.602 | 0.232 | 0.077 | 0.3386 | 20.92 | 1000 | 0 |

The best coarse retrieval result is achieved by `w16/s8/mean`, with
R@1 IoU 0.3 equal to 0.625. The best strict localization result is achieved by
`w8/s4/mean`, with R@1 IoU 0.5 equal to 0.393 and R@1 IoU 0.7 equal to 0.181.
The mIoU values of `w16/s8/mean` and `w8/s4/mean` are close: 0.3497 and 0.3478.

### 6.2 Smoothing Experiment

The smoothing experiment uses the same fixed 1,000-query subset and reuses the
existing embedding cache from the main sweep. The source file is
`results/charades_sta_smoothing_1000/summary.csv`.

| Configuration | Smoothing | R@1 IoU 0.3 | R@1 IoU 0.5 | R@1 IoU 0.7 | mIoU | Time (s) | Cache Hits | Cache Misses |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| CLIP w8/s4/mean | none | 0.509 | 0.393 | 0.181 | 0.3478 | 17.30 | 1000 | 0 |
| CLIP w8/s4/mean | moving_average_3 | 0.511 | 0.398 | 0.183 | 0.3485 | 15.72 | 1000 | 0 |
| CLIP w8/s4/mean | moving_average_5 | 0.509 | 0.398 | 0.190 | 0.3494 | 15.79 | 1000 | 0 |
| CLIP w16/s8/mean | none | 0.625 | 0.260 | 0.096 | 0.3497 | 14.79 | 1000 | 0 |
| CLIP w16/s8/mean | moving_average_3 | 0.620 | 0.257 | 0.098 | 0.3491 | 15.37 | 1000 | 0 |
| CLIP w16/s8/mean | moving_average_5 | 0.623 | 0.254 | 0.098 | 0.3481 | 16.00 | 1000 | 0 |

Smoothing slightly improves the stricter localization metrics for the shorter
`w8/s4/mean` configuration. The moving-average-5 variant reaches R@1 IoU 0.7
equal to 0.190 and mIoU equal to 0.3494. For the `w16/s8/mean` configuration,
smoothing does not improve the main coarse metric or mIoU; the unsmoothed
variant remains strongest at R@1 IoU 0.3 and mIoU.

### 6.3 CLIP vs Moment-DETR on the Same 50-Query Subset

The CLIP vs Moment-DETR comparison uses the same 50 Charades-STA query-video
pairs from:

```text
data/processed/charades_sta_moment_detr_test_subset.jsonl
```

The Moment-DETR result is a raw-video feasibility probe using the compatible
raw-video checkpoint from the external repository. It is not a full official
Moment-DETR benchmark and it is not fine-tuned on Charades-STA. The source files
are `results/clip_vs_moment_detr_50/comparison_summary.csv` and
`results/moment_detr_charades_50/metrics.json`.

| Model | Configuration | Queries | Failed | R@1 IoU 0.3 | R@1 IoU 0.5 | R@1 IoU 0.7 | mIoU | Time (s) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| CLIP | w8/s4/mean | 50 | 0 | 0.580 | 0.520 | 0.320 | 0.4394 | 14.22 |
| CLIP | w16/s8/mean | 50 | 0 | 0.680 | 0.360 | 0.080 | 0.4008 | 2.12 |
| CLIP | w32/s16/mean | 50 | 0 | 0.540 | 0.000 | 0.000 | 0.2854 | 2.09 |
| CLIP | w16/s8/max | 50 | 0 | 0.560 | 0.360 | 0.060 | 0.3375 | 2.09 |
| Moment-DETR | raw-video checkpoint | 50 | 0 | 0.440 | 0.320 | 0.040 | 0.2634 | 17.93 |

On this limited subset, the simple CLIP baseline performs strongly. The
`w8/s4/mean` CLIP configuration gives the best strict localization metrics,
while `w16/s8/mean` gives the best coarse metric at IoU 0.3. Moment-DETR
successfully produces predictions for all 50 examples, which confirms
raw-video feasibility, but this comparison is too small and too setup-specific
to support broad claims about model superiority.

## 7. Discussion

The experiments show a consistent trade-off between coarse retrieval and strict
localization. Medium 16-second windows perform best at IoU 0.3, where approximate
overlap is sufficient. Shorter 8-second windows perform better at stricter IoU
thresholds, because they can align more tightly with Charades-STA moments.
Large 32-second windows lose temporal precision and perform poorly at IoU 0.5
and 0.7.

Mean aggregation is more stable than max aggregation in the tested settings.
For the same `w16/s8` temporal setup, mean aggregation improves all reported
metrics compared with max aggregation in the 1,000-query sweep. This suggests
that the average visual-language relevance over a window is a better signal
than a single maximum-scoring frame for this dataset and baseline.

Similarity smoothing has a modest effect. It slightly improves strict
localization for the shorter `w8/s4/mean` setup, especially at IoU 0.7. However,
it does not improve the stronger coarse-retrieval configuration `w16/s8/mean`.
This suggests that smoothing may be useful when the candidate windows are short
and sensitive to local frame-score noise, but it is not a universal improvement.

The embedding cache is essential for practical experimentation. Without the
cache, each configuration would repeatedly run the CLIP image encoder on the
same videos. With caching, the expensive video encoding step is reused, making
controlled sweeps feasible on CPU.

The Moment-DETR probe demonstrates that integration with a specialized temporal
localization model is feasible, but sensitive to checkpoint and feature format.
The raw-video-compatible checkpoint works with local Charades videos. The
released QVHighlights feature checkpoint is incompatible with the raw-video
path due to feature dimension mismatch. Therefore, a rigorous Moment-DETR
benchmark would require a more careful environment and feature setup than the
limited probe performed here.

## 8. Limitations

The main CLIP experiment uses a fixed 1,000-query subset rather than the full
Charades-STA test split. The subset is large enough for a controlled coursework
experiment, but it should not be reported as the full official Charades-STA
benchmark.

All reported runs use CPU inference. Runtime may differ substantially on GPU or
with optimized batching.

The CLIP baseline uses pretrained CLIP only. No model is trained or fine-tuned
on Charades-STA. The method also does not explicitly model temporal action
dynamics beyond fixed windows, aggregation, and optional smoothing.

The Moment-DETR comparison uses only 50 examples and the raw-video-compatible
checkpoint from the external repository. It is a preliminary feasibility
comparison, not a full official Moment-DETR evaluation. Full official
benchmarking, feature extraction, and fine-tuning are outside the scope of this
coursework.

QVHighlights remains preliminary in this project because raw video availability
made it difficult to build a reproducible local experiment pipeline within the
available time.

## 9. Conclusion

This coursework implemented a reproducible raw-video Video Moment Retrieval
pipeline and evaluated it on Charades-STA. The final dataset setup contains
local Charades videos and standard Charades-STA annotations, avoiding the raw
video availability issues encountered with QVHighlights.

The main implemented method is a CLIP-based temporal-window retrieval baseline.
On a fixed 1,000-query Charades-STA subset, `w16/s8/mean` achieves the best
coarse retrieval result with R@1 IoU 0.3 equal to 0.625, while `w8/s4/mean`
achieves the best strict localization results with R@1 IoU 0.5 equal to 0.393
and R@1 IoU 0.7 equal to 0.181. Smoothing provides small improvements for the
short-window setup but does not consistently improve all configurations.

The project also demonstrates an isolated Moment-DETR raw-video probe on 50
Charades-STA examples. This confirms that a specialized temporal model can be
connected to the local data, but a full official benchmark and fine-tuning are
left outside the project scope.

Overall, the project provides a clear baseline, reusable data preparation
workflow, controlled experiment outputs, and a practical foundation for future
work on trained temporal localization models.

## 10. References and Project Evidence

Core references used by the project:

- Radford et al. introduced CLIP as a contrastive image-text pretraining model.
- Lei et al. introduced Moment-DETR and the QVHighlights benchmark for moment
  retrieval and highlight detection.
- Sigurdsson et al. introduced the Charades video dataset.
- Gao et al. introduced Charades-STA for temporal activity localization via
  language query.

BibTeX entries are collected in `thesis/references.bib`.

Current project evidence and result files used in this draft:

- `docs/moment_detr.md`
- `results/charades_sta_sweep_1000/summary.csv`
- `results/charades_sta_smoothing_1000/summary.csv`
- `results/clip_vs_moment_detr_50/comparison_summary.csv`
- `results/moment_detr_charades_50/metrics.json`

Intermediate writing notes were moved to local ignored agent memory and are
not part of the final coursework repository structure.
