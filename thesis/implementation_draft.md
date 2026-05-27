# Implementation

This chapter describes the current implementation state of the project. At this stage, the repository contains the main infrastructure for a reproducible Video Moment Retrieval pipeline, together with dummy and rehearsal pipelines used for sanity checks. The final CLIP-based baseline has not yet been executed; the current implementation is intended to make the later integration of CLIP and real video processing controlled and testable.

## Project Structure

The repository is organized as a small research pipeline rather than a monolithic script. The main source code is placed under `src/`, while runnable scripts, configuration files, experiment outputs, and tests are separated into dedicated directories.

The `src/data` package contains utilities for loading annotation files. The `src/evaluation` package implements temporal localization metrics. The `src/retrieval` package contains temporal window generation, score aggregation, smoothing, and best-window selection. The `src/video` package currently contains lightweight frame timestamp sampling utilities, without reading actual video files. The `src/pipeline` package contains end-to-end pipeline variants that connect data samples, retrieval logic, and evaluation.

The `scripts` directory contains command-line entry points for inspecting annotations, running dummy or rehearsal retrieval pipelines, collecting result files, and running small experiment workflows. The `configs` directory stores planned experiment settings, including a configuration draft for the future CLIP baseline. The `results` directory is used for JSON outputs and CSV summaries. The `tests` directory contains unit tests for the implemented modules and scripts.

## Evaluation Module

The evaluation layer implements the basic temporal metrics required for Video Moment Retrieval experiments. A prediction and a ground-truth moment are represented as temporal intervals in the format `[start, end]`. The temporal Intersection over Union is computed as the ratio between the length of the intersection and the length of the union of two intervals.

Since a query may have more than one relevant ground-truth window, the implementation also supports maximum temporal IoU over a list of ground-truth intervals. For each prediction, the best matching ground-truth interval is used.

The implemented retrieval metrics include `Recall@1` at IoU thresholds 0.3, 0.5, and 0.7. A prediction is counted as correct if its maximum IoU with the ground-truth windows is greater than or equal to the selected threshold. The module also implements mean IoU, computed as the average best IoU over all evaluated samples.

This evaluation module is independent of the retrieval method. It can be used with dummy predictions, frame-score based predictions, or future CLIP-based predictions.

## Annotation Loading

The project defines a `MomentRetrievalSample` data structure for representing a single Video Moment Retrieval example. Each sample contains a sample identifier, a video identifier, a text query, the video duration, and one or more relevant temporal windows.

The annotation loader is designed for QVHighlights-style files and supports both JSON and JSONL inputs. This is useful because dataset releases may differ in how annotation files are packaged. The loader also supports several alternative field names for key attributes such as sample id, video id, query text, duration, and relevant windows.

The loader performs basic validation before returning samples. It checks that required fields are present, that video duration can be converted to a positive floating-point value, that relevant windows are not empty, that every window has the `[start, end]` format, and that each window satisfies `end >= start`. This validation is important because later retrieval and evaluation stages assume a consistent temporal representation.

The annotation loader was validated on the official QVHighlights validation annotation file from the Moment-DETR repository. The file contains 1549 JSONL records. The real fields include `qid`, `query`, `duration`, `vid`, `relevant_clip_ids`, `saliency_scores`, and `relevant_windows`. The current implementation uses `qid`, `query`, `duration`, `vid`, and `relevant_windows`, mapping them to the internal sample representation. Saliency scores and relevant clip identifiers are not used in the current moment retrieval baseline skeleton.

## Temporal Window and Frame Sampling

The retrieval layer includes temporal window generation. Given a video duration, window size, and stride, the implementation generates candidate intervals fully contained inside the video timeline. If the regular striding pattern does not end exactly at the video duration, the final window is adjusted so that it ends at the duration. This makes the set of candidate windows cover the full video.

The project also includes lightweight frame timestamp sampling. Given a video duration and a target frames-per-second value, it generates timestamps in the interval `[0, duration)`. This does not decode video files or load images. It only defines the temporal positions at which frames would be sampled later.

These utilities are needed for the planned CLIP-based baseline. In that baseline, frames will be extracted at fixed timestamps, encoded with a CLIP image encoder, compared with a CLIP text embedding, and then aggregated into temporal windows.

## Score Aggregation and Retrieval

The retrieval utilities support the conversion from frame-level scores to window-level scores. A frame-level score represents how relevant a sampled frame is to a query. For each temporal window, the implementation selects frame scores whose timestamps fall inside that window and aggregates them.

Two aggregation methods are currently implemented: mean and max. Mean aggregation estimates the average relevance of frames inside a window, while max aggregation focuses on the most relevant frame inside the window. If a temporal window contains no frame timestamps, its score is set to `0.0`.

The project also implements optional score smoothing using a simple moving average. This is intended to reduce noise in frame-level scores before window aggregation. After window scores are computed, the best window is selected as the first window with the maximum score. This deterministic tie-breaking makes experiments reproducible.

## Rehearsal Pipelines

Two sanity-check pipelines are currently implemented.

The dummy dataset retrieval pipeline uses artificial window scores. It generates temporal windows for each sample, assigns deterministic scores based on the distance between each window center and the video center, selects the best window, and evaluates the result.

The frame-score rehearsal pipeline is closer to the planned CLIP baseline. It first generates frame timestamps, assigns artificial frame scores based on the distance between each timestamp and the video center, optionally smooths these scores, aggregates them into temporal windows, selects the best window, and computes metrics.

These pipelines are not final model results. They do not use CLIP, do not read video files, and do not measure real visual-language retrieval quality. Their purpose is to verify that the data representation, temporal windowing, score aggregation, prediction format, and evaluation logic work together before adding a real vision-language model.

## Experiment Workflow

The repository includes several command-line scripts for reproducible experimentation. The annotation inspection script loads an annotation file and prints basic information about the samples, durations, queries, and relevant windows. The dummy retrieval script runs the window-level sanity-check pipeline. The frame-score retrieval script runs the frame-score rehearsal pipeline with configurable FPS, window size, stride, aggregation method, smoothing, and sample limit.

Each pipeline writes a structured JSON result file containing configuration, predictions, ground-truth windows, metrics, and, when applicable, computational statistics such as the number of processed frame timestamps. A separate result collection script reads multiple JSON result files and writes a CSV summary. This CSV format is convenient for comparing experiments in the course report.

The repository also includes sample experiment scripts. One script runs a toy experiment matrix on synthetic sample annotations. Another script is prepared for real QVHighlights annotation files, assuming they have been added manually to the expected local directory. The latter script does not download data, read video, or run CLIP; it only checks the annotation-driven rehearsal workflow.

Annotation-only rehearsal experiments were run on the first 100 validation samples from the official QVHighlights validation annotations. These experiments verified that the loader, temporal window generation, frame-score rehearsal pipeline, evaluation code, and result collection work together on real annotation records. However, they do not represent final retrieval quality, because the frame scores are artificial and no visual features or CLIP embeddings are used.

A one-video CLIP retrieval smoke test was performed on a short synthetic local video. The purpose of this test was to validate the integration between frame extraction, CLIP text/image encoding, frame-level similarity computation, temporal window aggregation, and best-window selection. Since the synthetic video has no ground-truth temporal annotations, this smoke test does not report Recall@1 or mIoU and should not be interpreted as benchmark performance.

## Synthetic CLIP Benchmark

A tiny synthetic benchmark was created to validate the complete CLIP-based retrieval pipeline. It contains three short synthetic videos with simple colored-square events and corresponding temporal annotations. Unlike the previous rehearsal pipelines, this benchmark uses real CLIP image-text similarity scores.

The benchmark achieved `R@1` at IoU 0.3 equal to 1.0000, `R@1` at IoU 0.5 equal to 0.6667, `R@1` at IoU 0.7 equal to 0.6667, and `mIoU` equal to 0.7778. These results should not be interpreted as QVHighlights performance because the dataset is synthetic and very small. The purpose is to verify the end-to-end implementation before running experiments on real videos.

## QVHighlights Subset CLIP Run

A first QVHighlights subset run was performed on two locally available validation clips. The pipeline used real CLIP image-text similarities and evaluated predicted temporal windows against the ground-truth `relevant_windows` from the validation annotations. The run produced `R@1` at IoU 0.3, 0.5, and 0.7 equal to 0.0000, and `mIoU` equal to 0.0606. Because the subset contains only two videos, this result should be treated as a technical validation and preliminary observation, not as a reliable benchmark estimate. The result suggests that the naive pretrained CLIP baseline may require careful window sizing, score aggregation, smoothing, and larger evaluation subsets.

## Preliminary QVHighlights Subset Sweep

A preliminary parameter sweep was performed on two locally available QVHighlights validation clips. The experiment varied temporal window size and aggregation method. All `Recall@1` values at IoU thresholds 0.3, 0.5, and 0.7 were 0.0000. However, `mIoU` varied across configurations, with the best value 0.0857 obtained by `window_size=16`, `stride=8`, and mean aggregation. Mean aggregation outperformed max aggregation in all tested settings on this subset. These results should not be interpreted as full QVHighlights benchmark performance because the subset is extremely small. The main value of this experiment is validating the experimental workflow and identifying directions for further tuning.

## Current Limitations

Real CLIP embeddings are integrated for smoke tests and small local subsets, but a full QVHighlights validation experiment has not been performed yet.

Real video decoding is implemented for local video files, but the project does not include full QVHighlights video storage or dataset-wide video processing.

Real QVHighlights subset evaluation has been performed only on two locally available validation clips. This is not enough to estimate benchmark performance.

The current result files are sanity checks, not final retrieval quality measurements. They demonstrate that the pipeline structure works, but they should not be interpreted as model performance on Video Moment Retrieval.

## Next Implementation Steps

The next step is to manually add real QVHighlights annotation files to `data/qvhighlights/` and run the annotation inspection script on the validation split. If the real annotation format differs from the expected schema, the loader should be adapted while keeping the internal `MomentRetrievalSample` representation stable.

After the annotation format is verified, the dummy and frame-score rehearsal pipelines can be run on a small subset, for example the first 100 validation samples. This will test the dataset-level experiment workflow without requiring video decoding.

The next major implementation step is real frame extraction from video files. Once frames can be extracted at the timestamps generated by the sampling module, a pretrained CLIP model can be integrated to compute visual and textual embeddings. These embeddings should be cached to avoid repeated computation across experiments.

Finally, the first CLIP-based baseline should be run on a small subset of QVHighlights. The existing retrieval and evaluation modules can then be reused to compare frame-level retrieval, temporal windows, aggregation strategies, smoothing, and computational cost.
