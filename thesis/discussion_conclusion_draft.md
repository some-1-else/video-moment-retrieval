# Discussion and Conclusion

## Discussion

The implemented system demonstrates that a CLIP-based video moment retrieval pipeline can be built in a reproducible and modular way. The pipeline uses a pretrained CLIP ViT-B/32 model without additional training or fine-tuning. It supports frame sampling, CLIP image-text similarity computation, temporal window generation, score aggregation, best-window selection, evaluation metrics, result collection, and image embedding caching.

The experiments on the readable QVHighlights subset show limited but non-zero retrieval quality. The best `mIoU` was 0.2316 for 32-second windows with 16-second stride and mean aggregation. The best `R@1` at IoU 0.3 was 0.3333, achieved by both 16-second and 32-second mean-window configurations. These results are preliminary because the subset contains only 9 readable validation clips, so they should not be interpreted as full QVHighlights benchmark performance.

The results suggest that temporal window size and aggregation strategy matter. Mean aggregation consistently outperformed max aggregation for all tested window sizes. Longer windows performed better than shorter windows on this subset, which suggests that the simple CLIP baseline benefits from broader temporal context. However, strict localization remains difficult. The best `R@1` at IoU 0.7 was only 0.1111, indicating that naive frame-level CLIP similarity is not sufficient for precise moment boundary prediction.

This outcome is consistent with the expected limitations of a simple pretrained image-text model in a temporal retrieval task. CLIP can provide useful semantic similarity signals, but it does not explicitly model actions over time or learn temporal boundaries. Stronger performance on Video Moment Retrieval likely requires specialized temporal models, learned boundary prediction, or additional video-language training.

## Computational Cost

Repeated CLIP image encoding dominates the runtime of parameter sweeps. Without caching, the readable-subset sweep recomputed the same frame embeddings for every tested configuration. The uncached sweep took 332.5960 seconds in total.

To reduce this cost, a simple image embedding cache was implemented. The cache stores CLIP image embeddings per video, FPS value, and model name. With the cache enabled, the same sweep took 20.7162 seconds. This corresponds to approximately 16.05x speedup. The cache size was 2,591,373 bytes, or about 2.6 MB, for 9 clips and 1356 sampled frames.

This result shows that caching is important for reproducible experimentation. It allows window size, stride, and aggregation settings to be changed without repeatedly running the expensive CLIP image encoder on the same frames.

## Limitations

The current evaluation is limited by the small subset size. Only 9 readable QVHighlights validation clips were used, so the reported metrics are not representative of the full validation split.

Data availability is also a practical limitation. Some YouTube videos are unavailable or private, and one downloaded local file was unreadable by OpenCV. These issues make dataset construction and reproducibility more difficult.

The experiments were performed on CPU only, which limits throughput. The baseline also uses pretrained CLIP without fine-tuning, does not include a learned temporal boundary model, and does not explicitly model temporal action dynamics.

Finally, the project does not yet include a comparison with specialized moment retrieval models such as Moment-DETR or Lighthouse. Such a comparison would be useful for understanding the quality gap between a simple CLIP baseline and dedicated temporal video-language models.

## Future Work

Future work should first expand the readable QVHighlights subset to obtain more reliable estimates of retrieval quality. The video download and validation workflow can also be improved to better handle unavailable, private, corrupted, or codec-incompatible videos.

The embedding cache should be extended and used systematically for larger experiments. Text embedding caching may also be useful when the same queries are evaluated repeatedly. Additional FPS values should be tested to understand the trade-off between temporal resolution, quality, and computational cost.

Further retrieval experiments should include smoothing of frame-level similarity scores, additional window sizes and strides, and visualizations of score curves with predicted and ground-truth windows. These visualizations would help explain failure cases and understand how CLIP scores behave over time.

If time allows, the CLIP baseline should be compared with a specialized model such as Moment-DETR or Lighthouse. Such a comparison would clarify the trade-off between implementation simplicity, computational cost, and retrieval quality.

## Conclusion

The goal of this project was to build a clear and reproducible baseline for Video Moment Retrieval and to analyze both retrieval quality and computational cost. This goal was achieved at the level of a working CLIP-based pipeline on real QVHighlights clips.

The experiments show that temporal windowing and aggregation are important design choices. Mean aggregation and longer windows performed better on the current readable subset, while strict temporal localization remained challenging.

The computational experiments show that embedding caching is essential for efficient parameter sweeps. Caching reduced total sweep time from 332.5960 seconds to 20.7162 seconds while preserving identical retrieval metrics.

Overall, the implemented baseline provides a useful foundation for further experiments. It is not a state-of-the-art moment retrieval model, but it offers a reproducible starting point for studying the strengths and limitations of pretrained vision-language models in temporal video retrieval.
