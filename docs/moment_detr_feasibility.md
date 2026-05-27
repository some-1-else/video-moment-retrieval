# Moment-DETR Integration Feasibility

This note summarizes a lightweight investigation of whether Moment-DETR can be added as a comparison baseline for the current Video Moment Retrieval coursework.

## Context

Moment-DETR is a specialized temporal video-language model introduced with the QVHighlights dataset. In contrast to the current CLIP baseline, which computes frame-level image-text similarity and aggregates scores into temporal windows, Moment-DETR is designed to predict temporal moment coordinates and saliency scores directly from video and query representations.

For this project, Moment-DETR would be useful as a stronger published baseline. However, the goal of the current implementation is to keep the pipeline reproducible, lightweight, and focused on the CLIP baseline unless comparison can be added without major preprocessing or engineering cost.

Official repository:

```text
https://github.com/jayleicn/moment_detr
```

## What the Official Repository Provides

The official repository provides:

- QVHighlights annotation files;
- Moment-DETR model code;
- training scripts;
- inference scripts;
- evaluation utilities;
- released checkpoints for QVHighlights;
- a raw-video demo under `run_on_video/`.

The repository README states that it supports pre-training, fine-tuning, evaluation on QVHighlights, and prediction on custom raw videos.

## Checkpoints

The official GitHub releases include QVHighlights checkpoints:

- ASR pretrained checkpoint: `pt_model_e50.ckpt`;
- fine-tuned model from the pretrained checkpoint: `ft_model_from_pt_model_e50.ckpt`;
- model trained from scratch: `scratch_model.ckpt`.

Therefore, a pretrained/fine-tuned checkpoint appears to be available. This means inference without training should be possible in principle.

## Inference Script

The official repository includes an inference script:

```bash
bash moment_detr/scripts/inference.sh CHECKPOINT_PATH SPLIT_NAME
```

The script calls `moment_detr/inference.py` with:

```text
--resume CHECKPOINT_PATH
--eval_split_name SPLIT_NAME
--eval_path data/highlight_<split>_release.jsonl
```

This is dataset-level inference for the official QVHighlights-style setup.

## Expected Input Features

The main QVHighlights inference/training pipeline expects pre-extracted features rather than raw video frames.

The official README asks users to download:

```text
moment_detr_features.tar.gz
```

This archive is approximately 8GB. It contains pre-extracted features for the QVHighlights setup. The README says the training setup uses SlowFast and OpenAI CLIP features. The dataset loader expects `.npz` feature files for videos and text queries:

- video features from one or more `v_feat_dirs`;
- query/text features from `t_feat_dir`;
- video feature files named by `vid`;
- query feature files named by `qid`.

The model can also add temporal endpoint features (`tef`) to the video representation.

This is a major difference from the current project pipeline. Our CLIP baseline extracts frames from local videos and computes CLIP image embeddings directly. It does not use the official Moment-DETR feature archive or text feature directory format.

## Raw-Video Demo

The repository also includes a raw-video demo:

```bash
PYTHONPATH=$PYTHONPATH:. python run_on_video/run.py
```

This path uses a `MomentDETRPredictor`, a CLIP-based feature extractor, and a Moment-DETR checkpoint to run prediction on an example video and query file. The demo extracts video and text features inside the script.

This is more relevant to our readable subset because we have local `.mp4` clips. However, using it for our experiments would still require:

- downloading a Moment-DETR checkpoint;
- setting up the Moment-DETR environment;
- adapting the demo from one example video to multiple QVHighlights samples;
- converting Moment-DETR predictions into our existing metric format;
- making sure CPU execution works acceptably or configuring GPU execution;
- checking compatibility with our local video clips and their durations.

The demo also contains a constraint tied to the pretrained positional embedding: it supports up to 75 two-second clips, i.e. approximately 150 seconds. This likely matches many QVHighlights clips, but it is still a constraint that needs to be checked.

## Can We Run Inference on QVHighlights Without Training?

In principle, yes:

- official checkpoints are released;
- official inference script exists;
- official annotations are available.

In practice, dataset-level inference with the official script expects the official pre-extracted features. Using that route would require downloading the large `moment_detr_features.tar.gz` archive, which is outside the current lightweight scope.

The raw-video demo route may avoid the 8GB feature archive, but it is not a drop-in benchmark runner. It would need adaptation before it can evaluate our 9 readable QVHighlights clips with the same metrics as the CLIP baseline.

## Applying Moment-DETR to Our Readable Subset

Current subset:

```text
9 readable QVHighlights validation clips
local .mp4 files
queries and relevant_windows from validation annotations
```

Possible integration routes:

1. Official dataset-level inference:
   - requires official pre-extracted video and text features;
   - requires checkpoint;
   - closest to official evaluation;
   - not lightweight because of the 8GB feature archive.

2. Raw-video demo adaptation:
   - uses local `.mp4` videos;
   - requires checkpoint and Moment-DETR dependencies;
   - requires adapting the demo to batch over our manifest samples;
   - requires converting predictions to our metrics;
   - less storage-heavy than the 8GB feature archive, but still non-trivial.

3. Cite Moment-DETR as related work only:
   - no engineering risk;
   - appropriate for the current stage;
   - keeps focus on the implemented CLIP baseline.

## Risks

- The official dataset-level route requires a large feature archive.
- The official code targets older dependencies, including Python 3.7 and PyTorch 1.9.0 in the README.
- Running the raw-video demo may require separate environment setup.
- The demo defaults to CUDA-style usage and may need changes for CPU-only execution.
- Adapting predictions to our evaluation format requires extra code.
- Comparing raw-video-demo outputs with our CLIP baseline may not be fully equivalent to official QVHighlights evaluation.
- Downloading checkpoints is smaller than downloading features, but still an additional external artifact.
- This integration could distract from completing and writing up the current reproducible CLIP baseline.

## Feasibility Assessment

Moment-DETR comparison is technically possible, but not lightweight enough to add immediately without risk.

For a full official comparison, the main blocker is the pre-extracted feature archive. For a small local subset comparison, the raw-video demo is more promising, but it still requires checkpoint setup, environment compatibility work, script adaptation, and metric conversion.

Given the current timeframe and project scope, Moment-DETR should not be integrated into the main pipeline right now.

## Recommendation

Recommended choice: **cite as related work and leave implementation as future work**.

More specifically:

- do not add Moment-DETR code into `src/` at this stage;
- do not download the 8GB feature archive;
- do not spend time reproducing training or full official inference;
- mention Moment-DETR in Related Work as a specialized temporal model;
- describe comparison with Moment-DETR or Lighthouse as future work;
- if extra time remains, try a separate isolated raw-video demo experiment, not a core pipeline dependency.

This keeps the coursework focused and honest: the implemented contribution is the reproducible CLIP-based baseline, while Moment-DETR is a relevant specialized model for comparison in future work.
