# Results Summary

## Dataset Subset

The current CLIP-based experiments use QVHighlights validation annotations and a small locally available readable subset of videos. The subset contains 9 readable validation clips. At 1 FPS, this corresponds to 1356 processed frames.

All experiments in this summary use pretrained CLIP ViT-B/32 and CPU inference. The model is not fine-tuned. These results are preliminary subset results, not a full QVHighlights validation benchmark.

## CLIP Baseline Sweep

| setting | R@1@0.3 | R@1@0.5 | R@1@0.7 | mIoU | inference_time_sec |
|---|---:|---:|---:|---:|---:|
| w4/s2 mean | 0.1111 | 0.1111 | 0.0000 | 0.1225 | 40.5729 |
| w4/s2 max | 0.0000 | 0.0000 | 0.0000 | 0.0159 | 41.1622 |
| w8/s4 mean | 0.2222 | 0.1111 | 0.1111 | 0.1625 | 41.4672 |
| w8/s4 max | 0.0000 | 0.0000 | 0.0000 | 0.0316 | 41.7175 |
| w16/s8 mean | 0.3333 | 0.2222 | 0.1111 | 0.2271 | 40.6741 |
| w16/s8 max | 0.1111 | 0.0000 | 0.0000 | 0.0644 | 41.5106 |
| w32/s16 mean | 0.3333 | 0.2222 | 0.0000 | 0.2316 | 42.3688 |
| w32/s16 max | 0.2222 | 0.1111 | 0.0000 | 0.1584 | 43.1228 |

## Main Quality Observations

Mean aggregation outperformed max aggregation for all tested window sizes. Longer temporal windows performed better than short windows on this subset.

The best `mIoU` was 0.2316 for `w32/s16 mean`. The best `R@1@0.3` was 0.3333, achieved by both `w16/s8 mean` and `w32/s16 mean`.

Strict localization remains difficult. The best `R@1@0.7` was 0.1111, achieved by `w8/s4 mean` and `w16/s8 mean`.

## Computational Cost and Cache

| setting | total_time_sec | avg_time_sec | cache_size_bytes | speedup |
|---|---:|---:|---:|---:|
| uncached sweep | 332.5960 | 41.5745 | 0 | 1.00x |
| cached sweep | 20.7162 | 2.5895 | 2591373 | 16.05x |

The cached sweep reused CLIP image embeddings for the same videos, FPS, and model name. Retrieval metrics remained identical between the cached and uncached sweeps.

Caching is essential for efficient parameter sweeps because it avoids repeated CLIP image encoding when only temporal window size, stride, or aggregation changes.

## Limitations

- The subset contains only 9 clips.
- YouTube availability issues limited the number of videos.
- One downloaded local file was unreadable by OpenCV.
- Inference was performed on CPU only.
- The baseline uses pretrained CLIP without fine-tuning.
- The method does not include a temporal model.
- There is no comparison with Moment-DETR yet.
- The results are preliminary and should not be interpreted as full QVHighlights benchmark performance.

## Takeaway

The implemented CLIP baseline works end-to-end on real QVHighlights clips. Retrieval quality is limited but non-zero, and temporal window size and aggregation method clearly affect performance.

The embedding cache dramatically reduces repeated experiment cost while preserving identical retrieval metrics, making larger parameter sweeps much more practical.
