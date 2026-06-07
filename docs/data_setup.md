# Data Setup

This document describes the data preparation path needed to reproduce the final
coursework experiments: Charades-STA annotations plus local Charades videos.
The project does not automatically download the full Charades or Charades-STA
datasets.

## Expected Layout

Place Charades-STA annotation files here:

```text
data/charades_sta/
  charades_sta_train.txt
  charades_sta_test.txt
```

Place local Charades videos here:

```text
data/raw/charades/videos/
  <video_id>.mp4
```

The code also accepts `.avi` and `.mkv`, but `.mp4` is the preferred local
format. Each Charades-STA annotation line is expected to have this form:

```text
<video_id> <start_time> <end_time>##<sentence>
```

For example:

```text
ABC123 12.3 18.7##person opens a door
```

The matching local video path is:

```text
data/raw/charades/videos/ABC123.mp4
```

## Charades Video Archive

The raw Charades videos are stored in the official AI2 Charades archive. Keep
the downloaded archive outside git, for example:

```text
data/raw/charades/archives/Charades_v1_480.zip
```

For coursework experiments, the 480p archive is sufficient and easier to handle
locally than a larger version.

## Extract Videos

Start with a small subset, not the full archive:

```bash
.venv/bin/python scripts/prepare_data.py \
  --zip_path data/raw/charades/archives/Charades_v1_480.zip \
  --limit 20
```

The script extracts selected `.mp4` members into:

```text
data/raw/charades/videos/
```

and writes an extraction manifest:

```text
data/processed/charades_zip_extract_manifest.csv
```

If the required video ids are already known, put them in a text or CSV file:

```text
data/charades_sta/smoke_video_ids.txt
```

then extract only those videos:

```bash
.venv/bin/python scripts/prepare_data.py \
  --zip_path data/raw/charades/archives/Charades_v1_480.zip \
  --video_ids data/charades_sta/smoke_video_ids.txt \
  --limit 20
```

When both `--video_ids` and `--limit` are provided, the script first filters by
the requested ids and then extracts at most the first `N` matching files.

## Verify Data

Before running CLIP experiments, verify that annotations and local videos agree:

```bash
.venv/bin/python .agent_memory/scripts/probe_charades_sta.py \
  --annotations_path data/charades_sta/charades_sta_train.txt \
  --videos_dir data/raw/charades/videos \
  --n 20 \
  --output_manifest data/processed/charades_sta_probe_manifest.csv
```

The probe parses annotations and checks video readability with OpenCV. It does
not run CLIP, retrieval, or evaluation.

Successful rows should satisfy:

- `video_exists=True`
- `can_open=True`
- `can_read_first_frame=True`
- `window_within_duration=True`

Rows with missing videos, unreadable files, or out-of-duration annotations
should be excluded from the first smoke-test subset.

## Reproduction Notes

- Keep raw archives, extracted videos, manifests, caches, and embeddings out of
  git unless they are intentionally tiny sample files.
- Start with 5-20 verified videos before scaling to the fixed 1,000-query
  coursework subset.
- The final result folders keep their own `run_config.json`, `metrics.json`,
  `summary.csv`, and prediction files so that reported numbers can be checked
  without rerunning all inference.
