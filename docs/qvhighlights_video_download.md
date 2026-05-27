# QVHighlights Video Download

The project does not download the full QVHighlights video collection. Full-dataset video download is large, slow, brittle, and unnecessary before the CLIP pipeline has been validated on a small local subset.

The current download tooling is intentionally limited to a tiny subset manifest. It is meant for manual, controlled experiments with a few validation samples.

## Inputs

The downloader uses a manifest created by:

```bash
.venv/bin/python scripts/create_qvhighlights_subset_manifest.py \
  --annotations data/qvhighlights/highlight_val_release.jsonl \
  --videos-dir data/qvhighlights/videos \
  --limit 20 \
  --output data/qvhighlights/qvhighlights_val_subset_manifest.json
```

QVHighlights validation `video_id` values often follow this pattern:

```text
<youtube_id>_<clip_start>_<clip_end>
```

The downloader uses:

- `youtube_id` to locate the YouTube source video;
- `clip_start` and `clip_end` to trim the required segment;
- `video_id` to name the local output file.

The expected local output path is:

```text
data/qvhighlights/videos/<video_id>.mp4
```

## Dry Run

Always start with a dry run:

```bash
.venv/bin/python scripts/download_qvhighlights_subset_videos.py \
  --manifest data/qvhighlights/qvhighlights_val_subset_manifest.json \
  --output-dir data/qvhighlights/videos \
  --max-downloads 3 \
  --dry-run
```

This prints which videos would be downloaded and clipped. It does not access the network and does not write video files.

## Real Download

After checking the dry-run plan, remove `--dry-run`:

```bash
.venv/bin/python scripts/download_qvhighlights_subset_videos.py \
  --manifest data/qvhighlights/qvhighlights_val_subset_manifest.json \
  --output-dir data/qvhighlights/videos \
  --max-downloads 3
```

The script uses `yt-dlp` to download the source YouTube video into a temporary directory, then uses system `ffmpeg` to trim the requested segment. The temporary full source file is removed automatically.

## Dependencies

Python dependency:

```text
yt-dlp
```

System dependency:

```text
ffmpeg
```

If `ffmpeg` is not installed or not available on `PATH`, segment extraction will fail with a clear error message.

## Failure Modes

Some YouTube videos may be unavailable, region-restricted, removed, age-restricted, or blocked by network conditions. This is expected for web-hosted video datasets.

For reproducibility, keep the subset manifest used for an experiment together with the result metadata. The manifest records which samples were planned and which local video paths were expected.

## After Download

After downloading a few videos, recreate the manifest:

```bash
.venv/bin/python scripts/create_qvhighlights_subset_manifest.py \
  --annotations data/qvhighlights/highlight_val_release.jsonl \
  --videos-dir data/qvhighlights/videos \
  --limit 20 \
  --output data/qvhighlights/qvhighlights_val_subset_manifest.json
```

Then inspect it again and confirm that `num_available_videos` increased:

```bash
.venv/bin/python scripts/inspect_qvhighlights_subset_manifest.py \
  --manifest data/qvhighlights/qvhighlights_val_subset_manifest.json
```

Only after this should the project run CLIP retrieval on the available subset.
