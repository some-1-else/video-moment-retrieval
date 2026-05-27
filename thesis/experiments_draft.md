# Experiments and Results

## Experimental Setup

The current experimental evaluation uses the QVHighlights validation annotations. At this stage, the evaluation is restricted to a small readable local subset rather than the full validation split. The subset contains 9 locally available validation clips that passed video readability validation with OpenCV.

The retrieval model is based on the pretrained CLIP ViT-B/32 model. No training or fine-tuning is performed. Video frames are sampled at 1 FPS, encoded with the CLIP image encoder, and compared with the CLIP text embedding of the query using cosine similarity. The experiment processed 1356 frames in total and was executed on CPU.

The evaluated metrics are `Recall@1` at temporal IoU thresholds 0.3, 0.5 and 0.7, mean IoU (`mIoU`), and inference time. The reported inference time includes frame extraction, CLIP encoding, similarity computation, window aggregation, and evaluation overhead for each run.

## Tested Configurations

The sweep evaluates temporal window sizes of 4, 8, 16 and 32 seconds. The stride is set to half of the window size in each configuration:

- `window_size=4`, `stride=2`;
- `window_size=8`, `stride=4`;
- `window_size=16`, `stride=8`;
- `window_size=32`, `stride=16`.

For each temporal setup, two frame-score aggregation methods are tested: mean aggregation and max aggregation. Mean aggregation averages frame-level CLIP similarity scores inside each candidate window, while max aggregation uses the highest frame-level score inside the window.

## Results

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

## Discussion

Mean aggregation performed better than max aggregation across all tested window sizes. This suggests that, for this subset, averaging CLIP similarity over a temporal window provides a more stable signal than selecting a single highest-scoring frame.

Larger windows performed better than short windows on this small subset. The best `mIoU` was 0.2316, obtained by the 32-second window with 16-second stride and mean aggregation. However, the 16-second mean-window configuration achieved the same `R@1` at IoU 0.3 and better `R@1` at IoU 0.7. This indicates a trade-off between broader temporal coverage and stricter localization.

Strict temporal localization remains difficult. Even the best configurations achieve relatively low values at the IoU 0.7 threshold. This is expected for a naive pretrained CLIP baseline because CLIP scores are computed independently for sampled frames and do not explicitly model temporal action dynamics.

The results are preliminary and should not be interpreted as representative performance on the full QVHighlights validation split. The current subset contains only 9 readable clips. YouTube availability and local video decoding issues limited the subset size.

The main value of this experiment is to validate the experimental workflow on real QVHighlights clips and to identify promising directions for further tuning. In particular, the current results suggest that mean aggregation and longer temporal windows should be considered in the next round of experiments on a larger subset.

## Embedding Cache Experiment

Repeated sweeps over the same videos are inefficient if CLIP image embeddings are recomputed for every configuration. To reduce this cost, a simple `.npz` embedding cache was implemented for per-video image embeddings. The cache key depends on the `video_id`, CLIP `model_name`, and sampling `fps`. Text embeddings are not cached at this stage.

The cache was evaluated on the same 9 readable QVHighlights validation clips with `fps=1` and CLIP ViT-B/32. The first run populated the cache and produced 0 cache hits and 9 cache misses. The second run reused the cached embeddings and produced 9 cache hits and 0 cache misses.

| run | cache_hits | cache_misses | cache_size_bytes | inference_time_sec | R@1_IoU_0.3 | R@1_IoU_0.5 | R@1_IoU_0.7 | mIoU |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| first | 0 | 9 | 2591373 | 40.3804 | 0.3333 | 0.2222 | 0.1111 | 0.2271 |
| second | 9 | 0 | 2591373 | 2.8471 | 0.3333 | 0.2222 | 0.1111 | 0.2271 |

Inference time decreased from 40.38 seconds to 2.85 seconds, corresponding to approximately 14.2x speedup. The retrieval metrics remained unchanged, which confirms that the cache affects computational cost but not the retrieval output. The cache size was about 2.6 MB for 9 clips and 1356 sampled frames.

This result shows that caching is essential for efficient parameter sweeps. Once image embeddings are cached, different temporal window sizes, strides, and aggregation methods can be tested without repeatedly running the CLIP image encoder on the same frames.

## Cached Sweep and Computational Cost

The uncached sweep recomputed CLIP image embeddings for every configuration. This is inefficient because the same videos, model, and frame sampling rate are reused across all temporal window settings. The cached sweep reused per-video image embeddings from disk and evaluated the same 8 configurations as the uncached sweep.

Retrieval metrics were unchanged between the uncached and cached sweeps for every configuration. This confirms that caching changes only the computational cost, not the retrieval output.

| setting | total_time_sec | avg_time_sec | cache_size_bytes | speedup |
|---|---:|---:|---:|---:|
| uncached sweep | 332.5960 | 41.5745 | 0 | 1.00x |
| cached sweep | 20.7162 | 2.5895 | 2591373 | 16.05x |

Total inference time decreased from 332.60 seconds to 20.72 seconds. Average per-run inference time decreased from 41.57 seconds to 2.59 seconds. This corresponds to approximately 16.05x speedup. The embedding cache size was 2,591,373 bytes for 9 clips and 1356 sampled frames.

This result shows that embedding caching is essential for efficient experimentation with multiple temporal window settings. Without caching, each sweep repeatedly runs the CLIP image encoder on the same frames. With caching, the expensive visual encoding step is reused, making parameter sweeps substantially faster while preserving identical retrieval metrics.

## Limitations

- The experiment uses only 9 validation clips.
- Inference is performed on CPU.
- Text embeddings are not cached.
- The method does not include a temporal model.
- The baseline uses pretrained CLIP only.
- No training or fine-tuning is performed.
- Some YouTube videos are unavailable or private.
- Some downloaded local videos may be unreadable by OpenCV.
- No comparison with Moment-DETR or another specialized moment retrieval model has been performed yet.
