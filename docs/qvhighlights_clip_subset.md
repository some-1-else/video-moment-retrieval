# QVHighlights CLIP Subset Retrieval

This is the first planned CLIP retrieval workflow for real QVHighlights videos, but it is intentionally limited to a small local subset.

The script reads a subset manifest and runs CLIP only on samples whose local video file is already available:

```text
video_exists == true
```

It does not download missing videos, does not process the full validation split, and does not train or fine-tune any model.

## Command

```bash
.venv/bin/python scripts/run_qvhighlights_available_clip_retrieval.py \
  --manifest data/qvhighlights/qvhighlights_val_subset_manifest.json \
  --fps 1 \
  --window-size 8 \
  --stride 4 \
  --aggregation mean \
  --model-name ViT-B/32 \
  --output results/qvhighlights_available_clip_retrieval.json
```

The script:

- loads the manifest;
- filters only locally available samples;
- reconstructs `MomentRetrievalSample` objects;
- resolves the local `videos_dir`;
- runs the existing dataset-level CLIP retrieval pipeline;
- saves JSON output with config, predictions, ground truth, metrics, and stats.

## Expected Inputs

The manifest should be created after adding or downloading a small number of videos:

```bash
.venv/bin/python scripts/create_qvhighlights_subset_manifest.py \
  --annotations data/qvhighlights/highlight_val_release.jsonl \
  --videos-dir data/qvhighlights/videos \
  --limit 20 \
  --output data/qvhighlights/qvhighlights_val_subset_manifest.json
```

If no videos are available, the CLIP subset script exits with a clear message instead of trying to process missing files.

## Local video validation

File existence is not enough for a reliable CLIP run. Some downloaded `.mp4` files may exist on disk but still fail in OpenCV because the source download is incomplete, audio-only, corrupted, or encoded in a format that the local OpenCV build cannot decode.

Validate locally available videos before running CLIP retrieval on a larger subset:

```bash
.venv/bin/python scripts/validate_qvhighlights_videos.py \
  --manifest data/qvhighlights/qvhighlights_val_diverse50_subset_manifest.json \
  --output data/qvhighlights/qvhighlights_val_diverse50_video_validation.json
```

The validation report checks that each available file exists, can be opened by OpenCV, has positive duration/frame count, and can provide at least one frame.

Pass the validation report to CLIP retrieval:

```bash
.venv/bin/python scripts/run_qvhighlights_available_clip_retrieval.py \
  --manifest data/qvhighlights/qvhighlights_val_diverse50_subset_manifest.json \
  --validation-report data/qvhighlights/qvhighlights_val_diverse50_video_validation.json \
  --fps 1 \
  --window-size 16 \
  --stride 8 \
  --aggregation mean \
  --model-name ViT-B/32 \
  --output results/qvhighlights_available_clip_retrieval_diverse50_w16_s8_mean.json
```

When `--validation-report` is provided, unreadable videos are skipped. If no readable videos remain, the script exits with a clear message.

## Embedding cache

CLIP image embedding extraction is the most expensive repeated step in parameter sweeps. The retrieval scripts can optionally cache image embeddings per video, FPS, and CLIP model name. This means that later runs with different `window_size`, `stride`, or `aggregation` can reuse frame embeddings instead of recomputing them.

Only image embeddings are cached. Text query embeddings are not cached yet. Cache files are stored as `.npz` files, so the cache speeds up repeated experiments but uses additional disk space.

Enable cache with:

```bash
.venv/bin/python scripts/run_qvhighlights_available_clip_retrieval.py \
  --manifest data/qvhighlights/qvhighlights_val_diverse50_subset_manifest.json \
  --validation-report data/qvhighlights/qvhighlights_val_diverse50_video_validation.json \
  --fps 1 \
  --window-size 16 \
  --stride 8 \
  --aggregation mean \
  --model-name ViT-B/32 \
  --use-cache \
  --embeddings-cache-dir results/embeddings/qvhighlights_readable \
  --output results/qvhighlights_available_clip_retrieval_cached.json
```

## Limitations

This is not a full QVHighlights validation result. It measures behavior only on the videos that exist locally in the small subset manifest.

The metrics are useful for checking the real-video CLIP pipeline, but they should not be reported as benchmark performance for the full QVHighlights validation split.

## First available-subset run

The first available-subset run was executed with a diverse QVHighlights manifest and 2 locally available validation clips:

```bash
.venv/bin/python scripts/run_qvhighlights_available_clip_retrieval.py \
  --manifest data/qvhighlights/qvhighlights_val_diverse_subset_manifest.json \
  --fps 1 \
  --window-size 8 \
  --stride 4 \
  --aggregation mean \
  --model-name ViT-B/32 \
  --output results/qvhighlights_available_clip_retrieval_diverse.json
```

Manifest state before the run:

```text
available videos = 2
missing videos = 18
```

Metrics:

```text
R@1_IoU_0.3 = 0.0000
R@1_IoU_0.5 = 0.0000
R@1_IoU_0.7 = 0.0000
mIoU = 0.0606
```

Stats:

```text
num_samples = 2
total_frames = 302
avg_frames_per_sample = 151.0000
inference_time_sec = 11.4364
```

Output JSON:

```text
results/qvhighlights_available_clip_retrieval_diverse.json
```

Interpretation:

- the retrieval pipeline works end-to-end on real QVHighlights clips;
- the metrics are not representative because the subset contains only 2 videos;
- zero `Recall@1` indicates that the current naive pretrained CLIP baseline is not yet sufficient on this tiny subset;
- more videos, parameter sweeps, window-size tuning, aggregation variants, and smoothing experiments are needed before drawing conclusions.

## Available-subset parameter sweep

The available-subset sweep runs several CLIP retrieval configurations only on locally available QVHighlights videos from the diverse manifest. It does not download videos, does not process missing videos, and is not a full validation benchmark.

The goal is to check sensitivity to temporal window size and score aggregation:

- `window_size=4`, `stride=2`, `aggregation=mean`;
- `window_size=4`, `stride=2`, `aggregation=max`;
- `window_size=8`, `stride=4`, `aggregation=mean`;
- `window_size=8`, `stride=4`, `aggregation=max`;
- `window_size=16`, `stride=8`, `aggregation=mean`;
- `window_size=16`, `stride=8`, `aggregation=max`.

Run:

```bash
bash scripts/run_qvhighlights_available_sweep.sh
```

Outputs are saved under:

```text
results/qvhighlights_available_sweep/
```

The collected summary table is:

```text
results/qvhighlights_available_sweep/summary.csv
```

## Available-subset parameter sweep results

The available-subset parameter sweep was executed on 2 locally available QVHighlights validation clips. The output summary was saved to:

```text
results/qvhighlights_available_sweep/summary.csv
```

| result_file | window_size | stride | aggregation | R@1_IoU_0.3 | R@1_IoU_0.5 | R@1_IoU_0.7 | mIoU | total_frames | inference_time_sec |
|---|---:|---:|---|---:|---:|---:|---:|---:|---:|
| clip_w16_s8_max.json | 16.0 | 8.0 | max | 0.0 | 0.0 | 0.0 | 0.0625 | 302 | 11.4585 |
| clip_w16_s8_mean.json | 16.0 | 8.0 | mean | 0.0 | 0.0 | 0.0 | 0.0857 | 302 | 11.3162 |
| clip_w4_s2_max.json | 4.0 | 2.0 | max | 0.0 | 0.0 | 0.0 | 0.0147 | 302 | 11.1510 |
| clip_w4_s2_mean.json | 4.0 | 2.0 | mean | 0.0 | 0.0 | 0.0 | 0.0303 | 302 | 11.4622 |
| clip_w8_s4_max.json | 8.0 | 4.0 | max | 0.0 | 0.0 | 0.0 | 0.0286 | 302 | 11.2692 |
| clip_w8_s4_mean.json | 8.0 | 4.0 | mean | 0.0 | 0.0 | 0.0 | 0.0606 | 302 | 11.2406 |

Interpretation:

- `Recall@1` remains 0.0 for all tested settings;
- `mIoU` differs by configuration;
- the best observed `mIoU` is 0.0857 for `window_size=16`, `stride=8`, `aggregation=mean`;
- mean aggregation was better than max aggregation for all tested window sizes on this tiny subset;
- the results are not representative of full QVHighlights performance because only 2 samples were evaluated.

## Readable-subset parameter sweep

The readable-subset sweep uses the larger diverse-50 manifest together with the local video validation report. This means that downloaded files that exist on disk but cannot be opened by OpenCV are filtered out before CLIP retrieval starts.

The current validation report contains 9 readable QVHighlights validation clips and 1 unreadable local file. The sweep therefore evaluates only the 9 readable clips. It is still a preliminary subset experiment, not a full QVHighlights validation benchmark.

Run:

```bash
bash scripts/run_qvhighlights_readable_sweep.sh
```

The sweep tests `mean` and `max` aggregation for temporal windows of 4, 8, 16, and 32 seconds. Outputs are saved under:

```text
results/qvhighlights_readable_sweep/
```

The collected summary table is:

```text
results/qvhighlights_readable_sweep/summary.csv
```

## Readable-subset sweep results

The readable-subset parameter sweep was executed on 9 QVHighlights validation clips that passed local OpenCV readability validation. The sweep used real CLIP image-text similarities and did not download any new videos. The output summary was saved to:

```text
results/qvhighlights_readable_sweep/summary.csv
```

| result_file | window_size | stride | aggregation | R@1_IoU_0.3 | R@1_IoU_0.5 | R@1_IoU_0.7 | mIoU | total_frames | inference_time_sec |
|---|---:|---:|---|---:|---:|---:|---:|---:|---:|
| clip_w16_s8_max.json | 16.0 | 8.0 | max | 0.1111 | 0.0000 | 0.0000 | 0.0644 | 1356 | 41.5106 |
| clip_w16_s8_mean.json | 16.0 | 8.0 | mean | 0.3333 | 0.2222 | 0.1111 | 0.2271 | 1356 | 40.6741 |
| clip_w32_s16_max.json | 32.0 | 16.0 | max | 0.2222 | 0.1111 | 0.0000 | 0.1584 | 1356 | 43.1228 |
| clip_w32_s16_mean.json | 32.0 | 16.0 | mean | 0.3333 | 0.2222 | 0.0000 | 0.2316 | 1356 | 42.3688 |
| clip_w4_s2_max.json | 4.0 | 2.0 | max | 0.0000 | 0.0000 | 0.0000 | 0.0159 | 1356 | 41.1622 |
| clip_w4_s2_mean.json | 4.0 | 2.0 | mean | 0.1111 | 0.1111 | 0.0000 | 0.1225 | 1356 | 40.5729 |
| clip_w8_s4_max.json | 8.0 | 4.0 | max | 0.0000 | 0.0000 | 0.0000 | 0.0316 | 1356 | 41.7175 |
| clip_w8_s4_mean.json | 8.0 | 4.0 | mean | 0.2222 | 0.1111 | 0.1111 | 0.1625 | 1356 | 41.4672 |

Interpretation:

- mean aggregation consistently outperformed max aggregation for every tested window size;
- longer temporal windows gave better results than short windows on this small subset;
- the best `mIoU` was obtained by `window_size=32`, `stride=16`, `aggregation=mean`;
- `window_size=16`, `stride=8`, `aggregation=mean` and `window_size=32`, `stride=16`, `aggregation=mean` tied on `R@1` at IoU 0.3;
- `window_size=16`, `stride=8`, `aggregation=mean` achieved better `R@1` at IoU 0.7 than the 32-second mean-window configuration;
- the results are preliminary because only 9 readable clips were evaluated;
- inference time mostly reflects repeated CLIP encoding because embeddings are not cached yet.

## Embedding cache experiment

CLIP image embeddings are cached per `video_id`, `fps`, and `model_name`. Text embeddings are not cached yet, because query encoding is relatively cheap compared with repeatedly encoding all sampled video frames.

The first cached run fills the cache. The second run uses the same videos, FPS, model, and retrieval configuration, so it reuses cached image embeddings from disk.

| run | cache_hits | cache_misses | cache_size_bytes | inference_time_sec | R@1_IoU_0.3 | R@1_IoU_0.5 | R@1_IoU_0.7 | mIoU |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| first | 0 | 9 | 2591373 | 40.3804 | 0.3333 | 0.2222 | 0.1111 | 0.2271 |
| second | 9 | 0 | 2591373 | 2.8471 | 0.3333 | 0.2222 | 0.1111 | 0.2271 |

Interpretation:

- the cache gives approximately 14.2x speedup for repeated runs on the same videos;
- retrieval metrics are identical between the uncached and cached runs;
- the cache size is about 2.6 MB for 9 clips and 1356 sampled frames;
- this is important for parameter sweeps, because window size, stride, and aggregation can be changed without recomputing CLIP image embeddings.

## Cached readable-subset sweep

The cached readable-subset sweep uses the same 8 configurations as the uncached readable-subset sweep, but each run enables the image embedding cache:

```bash
bash scripts/run_qvhighlights_readable_sweep_cached.sh
```

The sweep reuses CLIP image embeddings from:

```text
results/embeddings/qvhighlights_readable
```

Outputs are saved under:

```text
results/qvhighlights_readable_sweep_cached/
```

The collected summary table is:

```text
results/qvhighlights_readable_sweep_cached/summary.csv
```

The summary includes `cache_hits`, `cache_misses`, and `embedding_cache_size_bytes`, which should be used to compare computational cost against the uncached sweep.

## Cached readable-subset sweep results

The cached readable-subset sweep was executed with the same 8 configurations as the uncached readable-subset sweep. The output summary was saved to:

```text
results/qvhighlights_readable_sweep_cached/summary.csv
```

| result_file | window_size | stride | aggregation | R@1_IoU_0.3 | R@1_IoU_0.5 | R@1_IoU_0.7 | mIoU | inference_time_sec | cache_hits | cache_misses | embedding_cache_size_bytes |
|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| clip_w16_s8_max.json | 16.0 | 8.0 | max | 0.1111 | 0.0000 | 0.0000 | 0.0644 | 2.5198 | 9 | 0 | 2591373 |
| clip_w16_s8_mean.json | 16.0 | 8.0 | mean | 0.3333 | 0.2222 | 0.1111 | 0.2271 | 2.5676 | 9 | 0 | 2591373 |
| clip_w32_s16_max.json | 32.0 | 16.0 | max | 0.2222 | 0.1111 | 0.0000 | 0.1584 | 2.5336 | 9 | 0 | 2591373 |
| clip_w32_s16_mean.json | 32.0 | 16.0 | mean | 0.3333 | 0.2222 | 0.0000 | 0.2316 | 2.5605 | 9 | 0 | 2591373 |
| clip_w4_s2_max.json | 4.0 | 2.0 | max | 0.0000 | 0.0000 | 0.0000 | 0.0159 | 2.5026 | 9 | 0 | 2591373 |
| clip_w4_s2_mean.json | 4.0 | 2.0 | mean | 0.1111 | 0.1111 | 0.0000 | 0.1225 | 2.9955 | 9 | 0 | 2591373 |
| clip_w8_s4_max.json | 8.0 | 4.0 | max | 0.0000 | 0.0000 | 0.0000 | 0.0316 | 2.5449 | 9 | 0 | 2591373 |
| clip_w8_s4_mean.json | 8.0 | 4.0 | mean | 0.2222 | 0.1111 | 0.1111 | 0.1625 | 2.4917 | 9 | 0 | 2591373 |

Computational-cost comparison:

```text
uncached total inference time = 332.5960s
cached total inference time = 20.7162s
speedup = approximately 16.05x
cache size = 2,591,373 bytes
```

The metrics were identical for all configurations between the uncached and cached sweeps. This confirms that the image embedding cache does not affect retrieval quality. The speedup comes from avoiding repeated CLIP image encoding for the same sampled frames, which makes the cache especially useful for parameter sweeps over temporal window size, stride, and score aggregation.
