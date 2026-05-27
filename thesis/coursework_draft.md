# Search for Video Moments Using Text Queries: Implementations and Experiments with Video-Language Retrieval Models

> Note: This is a draft assembled from intermediate thesis fragments and still requires citation cleanup, related work expansion, and formatting.

## Abstract

Video Moment Retrieval is the task of finding a temporal segment in a video that matches a natural-language query. This coursework studies a practical baseline for this task using pretrained video-language components rather than training a large model from scratch. The implemented system uses QVHighlights validation annotations, local video clips, frame sampling, CLIP image-text similarity, temporal window generation, score aggregation, and standard temporal retrieval metrics. The experiments are performed on a small readable subset of 9 QVHighlights validation clips, so the reported results are preliminary subset results rather than a full benchmark evaluation. The sweep compares different temporal window sizes, strides, and aggregation methods. Mean aggregation and longer windows perform better on the current subset, with the best `mIoU` of 0.2316. The work also evaluates computational cost and shows that caching CLIP image embeddings reduces repeated sweep time from 332.5960 seconds to 20.7162 seconds, a speedup of approximately 16.05x, while preserving identical metrics. Overall, the project provides a reproducible CLIP-based baseline and an experimental framework for further Video Moment Retrieval experiments.

## 1. Introduction

Video Moment Retrieval aims to localize the part of a video that corresponds to a text query. For example, given the query "a person opens a door", the system should return the start and end time of the relevant event in the video. This task is important for video search, video understanding, dataset exploration, and applications where users need to navigate long videos using natural language.

The task is more difficult than standard image-text retrieval because the output is not only a matching video or frame, but a temporal segment with boundaries. A query can describe an action that unfolds over several seconds, and relevant visual evidence may be distributed across multiple frames. Therefore, a system must connect text semantics with visual content and also make temporal localization decisions.

The goal of this coursework is to build a clear and reproducible baseline for text-query-based video moment retrieval. The focus is not on training a state-of-the-art model from scratch. Instead, the project studies a practical CLIP-based baseline that can be implemented, tested, and analyzed within a realistic course-project scope.

The baseline uses pretrained CLIP ViT-B/32 to compute image-text similarity between sampled video frames and a text query. Frame-level scores are aggregated into candidate temporal windows, and the best-scoring window is evaluated against ground-truth moment annotations. The experiments consider both retrieval quality and computational cost.

The current evaluation uses a small readable subset of QVHighlights validation clips. The results are therefore preliminary and should not be interpreted as full QVHighlights benchmark performance. Nevertheless, the experiments are useful for validating the pipeline and identifying the main trade-offs of a simple CLIP-based approach.

## 2. Related Work

### 2.1 Vision-Language Pretraining and CLIP

CLIP is a vision-language model trained with large-scale natural language supervision [1]. It learns a shared embedding space for images and text, where matching image-text pairs should be close to each other. This makes CLIP useful for zero-shot and retrieval-style tasks, because a text query can be compared directly with visual content without training a task-specific classifier.

In this coursework, CLIP is used as a simple pretrained baseline for text-to-video moment retrieval. Since CLIP is originally an image-text model, a video is represented as a sequence of sampled frames. Each frame is encoded independently, and the query is encoded with the CLIP text encoder. Frame-query similarity scores are then aggregated over temporal windows.

This approach is practical and reproducible, but it also has an important limitation: CLIP does not directly model temporal boundaries or action dynamics. Therefore, CLIP frame similarity can provide a semantic signal, but it is not expected to solve precise temporal localization by itself.

### 2.2 Video Moment Retrieval

Video Moment Retrieval is the task of localizing a temporal segment in a video given a natural-language query. Unlike video-level retrieval, where the system only needs to select the most relevant video, moment retrieval requires predicting start and end times inside a video.

This makes the task sensitive to temporal localization accuracy. A prediction can be semantically related to the query but still receive a low score if its boundaries do not overlap sufficiently with the ground truth. Standard evaluation therefore uses metrics based on temporal Intersection over Union, such as `Recall@1` at different IoU thresholds and mean IoU.

The current project follows this evaluation style. The predicted window is compared with one or more ground-truth relevant windows, and retrieval quality is reported at IoU thresholds 0.3, 0.5, and 0.7.

### 2.3 QVHighlights Dataset

QVHighlights is a benchmark for query-based video moment retrieval and highlight detection [2]. It combines natural-language queries with relevant temporal moments and saliency annotations. This makes it suitable for evaluating both whether a system can find query-relevant moments and whether it can identify especially salient parts of a video.

The real QVHighlights validation annotations used in this project contain fields such as `qid`, `query`, `duration`, `vid`, `relevant_clip_ids`, `saliency_scores`, and `relevant_windows`. In the current CLIP baseline, only the query, duration, video id, and `relevant_windows` are used. The saliency scores are not used because the implemented pipeline focuses on moment retrieval rather than highlight detection.

### 2.4 Specialized Temporal Models

Specialized temporal video-language models are designed to predict temporal spans more directly than a frame-level CLIP baseline. Moment-DETR, introduced together with QVHighlights, formulates moment retrieval as a direct set prediction problem and predicts moment coordinates and saliency scores end-to-end from video and query representations [2].

This coursework takes a different approach. Instead of reproducing or training a specialized temporal model, it implements a simple CLIP-based baseline. This choice keeps the scope realistic and allows the project to focus on reproducibility, evaluation, temporal windowing, aggregation, and computational cost.

Future work may include comparison with specialized systems such as Moment-DETR or Lighthouse. Such a comparison would help quantify the quality gap between a simple pretrained CLIP baseline and models designed specifically for temporal moment retrieval. Lighthouse is mentioned here as a possible future comparison target, but its exact reference details still need to be added.

## 3. Method

The project is implemented as a modular research pipeline. The main components are data loading, video frame sampling and extraction, CLIP encoding, temporal window generation, score aggregation, best-window selection, evaluation, result collection, and embedding caching.

The annotation loader reads QVHighlights-style JSON or JSONL annotation files. Each sample is represented as a `MomentRetrievalSample` with a sample id, video id, text query, video duration, and one or more ground-truth relevant windows. The loader supports several field-name variants and validates the temporal window format.

Video frames are sampled at a fixed FPS. For the current experiments, `fps=1` is used. Local videos are validated before retrieval because file existence is not enough: some downloaded files may be corrupted, audio-only, or unreadable by OpenCV. The readable-subset experiments use only videos that pass this validation step.

The retrieval baseline uses pretrained CLIP ViT-B/32. For each sample, the text query is encoded with the CLIP text encoder, and sampled frames are encoded with the CLIP image encoder. Cosine similarity between the normalized text embedding and image embeddings gives frame-level relevance scores.

Candidate temporal windows are generated with a fixed window size and stride. For example, `window_size=16` and `stride=8` produces overlapping 16-second windows. Frame-level scores are aggregated into each window using either mean aggregation or max aggregation. The window with the highest aggregated score is selected as the predicted moment.

The evaluation metrics are `Recall@1` at temporal IoU thresholds 0.3, 0.5, and 0.7, and mean IoU (`mIoU`). These metrics compare the predicted window with the ground-truth relevant windows for each query.

To reduce repeated computation, the project includes a simple image embedding cache. CLIP image embeddings are cached per `video_id`, `model_name`, and `fps` in `.npz` files. Text embeddings are not cached yet. The cache is especially useful for parameter sweeps where the videos and frame sampling remain fixed but window size, stride, and aggregation change.

## 4. Experiments

The current experiments use QVHighlights validation annotations and a small local readable subset of 9 validation clips. At 1 FPS, this subset contains 1356 processed frames. All experiments use pretrained CLIP ViT-B/32 on CPU, without training or fine-tuning.

The tested temporal window sizes are 4, 8, 16, and 32 seconds. The stride is set to half of the window size. For each temporal setup, two aggregation methods are evaluated: mean and max.

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

The best `mIoU` is 0.2316 for `w32/s16 mean`. The best `R@1@0.3` is 0.3333, achieved by both `w16/s8 mean` and `w32/s16 mean`. The best `R@1@0.7` is 0.1111, achieved by `w8/s4 mean` and `w16/s8 mean`.

The computational-cost comparison shows the effect of the image embedding cache:

| setting | total_time_sec | avg_time_sec | cache_size_bytes | speedup |
|---|---:|---:|---:|---:|
| uncached sweep | 332.5960 | 41.5745 | 0 | 1.00x |
| cached sweep | 20.7162 | 2.5895 | 2591373 | 16.05x |

The cached and uncached sweeps produce identical retrieval metrics. The speedup comes from avoiding repeated CLIP image encoding for the same sampled frames.

## 5. Discussion

The implemented CLIP-based retrieval pipeline works end-to-end on real QVHighlights clips. The quality is limited but non-zero, which is a reasonable outcome for a simple pretrained CLIP baseline without temporal training.

The results show that temporal windowing and aggregation strategy affect retrieval performance. Mean aggregation consistently outperforms max aggregation in the current subset. Longer temporal windows also perform better than shorter windows, suggesting that the baseline benefits from broader temporal context.

Strict temporal localization remains difficult. The best `R@1@0.7` is only 0.1111, which indicates that naive CLIP frame similarity is not enough to predict precise moment boundaries. This supports the expectation that stronger performance likely requires temporal modeling, learned boundary prediction, or specialized moment retrieval architectures.

Computationally, repeated CLIP image encoding is the main bottleneck for parameter sweeps. The embedding cache reduces total sweep time from 332.5960 seconds to 20.7162 seconds, about 16.05x faster, while leaving retrieval metrics unchanged.

The main limitations are the small subset size, YouTube video availability issues, one unreadable downloaded video, CPU-only inference, no fine-tuning, no learned temporal model, and no comparison with specialized models such as Moment-DETR or Lighthouse. Therefore, all reported QVHighlights results should be treated as preliminary subset results.

## 6. Conclusion

This coursework set out to build a reproducible baseline for Video Moment Retrieval and to analyze both retrieval quality and computational cost. The project achieved a working CLIP-based retrieval pipeline on real QVHighlights clips.

The experiments show that temporal window size and score aggregation matter. Mean aggregation and longer windows performed best on the current readable subset, while strict temporal localization remained challenging.

The project also shows that caching is essential for efficient experimentation. A simple image embedding cache preserved identical retrieval metrics while reducing repeated sweep time by approximately 16.05x.

Overall, the implemented baseline is not a state-of-the-art moment retrieval model, but it provides a clear and reproducible foundation for further experiments. Future work should expand the readable QVHighlights subset, add more robust video handling, test additional FPS values and smoothing methods, visualize score curves, and compare against specialized temporal models if time allows.

## References

[1] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen Krueger, and Ilya Sutskever. Learning Transferable Visual Models From Natural Language Supervision. ICML, 2021.

[2] Jie Lei, Tamara L. Berg, and Mohit Bansal. Detecting Moments and Highlights in Videos via Natural Language Queries. NeurIPS, 2021.

[3] TODO: Lighthouse paper details.

[4] TODO: Additional related work on text-to-video retrieval and temporal video grounding.
