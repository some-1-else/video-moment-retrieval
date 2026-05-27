# QVHighlights Video Subset

The full QVHighlights video collection is too large to introduce into the project as the first video experiment step. The current strategy is to work with a small local subset first: a few validation samples, only the corresponding local video files, and no dataset-wide CLIP run.

This keeps the workflow reproducible and debuggable while the project moves from annotation-only checks to real video-based CLIP retrieval.

Optional tiny-subset video download is documented in [docs/qvhighlights_video_download.md](qvhighlights_video_download.md). The downloader is not used automatically and should be tested with `--dry-run` first.

## Expected Layout

Local videos should be placed manually under:

```text
data/qvhighlights/videos/
```

The current tooling expects one video file per `video_id`:

```text
data/qvhighlights/videos/<video_id>.mp4
```

For example, if an annotation sample has `video_id = "example_123"`, the expected local path is:

```text
data/qvhighlights/videos/example_123.mp4
```

QVHighlights videos are not downloaded automatically. Missing videos are normal at this stage.

## QVHighlights video_id format

In the checked validation annotations, `video_id` values often look like:

```text
<youtube_id>_<clip_start>_<clip_end>
```

For example:

```text
NUsG9BgSes0_210.0_360.0
```

This can be parsed as:

```text
youtube_id = NUsG9BgSes0
clip_start = 210.0
clip_end = 360.0
```

The parser works from the end of the string, so YouTube IDs that contain underscores are still handled. If a `video_id` does not end with two numeric timestamp fields, it is kept as a plain ID and the parsed clip range is set to `null`.

The local video path is still expected to use the full `video_id`:

```text
data/qvhighlights/videos/<video_id>.mp4
```

The parsed `youtube_id`, `clip_start`, and `clip_end` fields are useful for a future manual or semi-automatic video preparation step. At the current stage, the project only parses these identifiers and does not download videos automatically.

## Create a Subset Manifest

After the validation annotations are available locally, create a manifest for the first `N` samples:

```bash
.venv/bin/python scripts/create_qvhighlights_subset_manifest.py \
  --annotations data/qvhighlights/highlight_val_release.jsonl \
  --videos-dir data/qvhighlights/videos \
  --limit 20 \
  --output data/qvhighlights/qvhighlights_val_subset_manifest.json
```

The manifest records:

- the selected annotation samples;
- parsed `youtube_id`, `clip_start`, and `clip_end` fields;
- the expected local video path for each sample;
- whether each local `.mp4` file currently exists;
- the number of available and missing videos.

Because the manifest depends on local files, it is ignored by git.

## Subset Selection Strategies

The manifest script supports two deterministic strategies:

`first`: takes the first `N` annotation samples. This is simple, but it can choose many clips from the same YouTube source. In practice, the first 20 validation samples pointed to one unavailable YouTube video, so all download attempts failed.

`diverse-youtube`: walks through the annotation file from top to bottom and takes at most one sample per parsed `youtube_id`. This gives download attempts a better chance because the subset covers more distinct source videos.

Create a diverse manifest:

```bash
.venv/bin/python scripts/create_qvhighlights_subset_manifest.py \
  --annotations data/qvhighlights/highlight_val_release.jsonl \
  --videos-dir data/qvhighlights/videos \
  --limit 20 \
  --strategy diverse-youtube \
  --output data/qvhighlights/qvhighlights_val_diverse_subset_manifest.json
```

## Inspect the Manifest

To inspect the local availability status:

```bash
.venv/bin/python scripts/inspect_qvhighlights_subset_manifest.py \
  --manifest data/qvhighlights/qvhighlights_val_subset_manifest.json
```

The script prints the number of samples, available videos, missing videos, several missing and available `video_id` values, and a few query/window examples.

## Next Step

Once a few matching videos are placed in `data/qvhighlights/videos/`, the next step is to run CLIP retrieval only on samples whose `video_exists` value is `true`. This will allow a small real-video QVHighlights subset experiment without downloading or processing the full dataset.

If videos are downloaded manually or with the optional subset downloader, recreate the manifest afterward:

```bash
.venv/bin/python scripts/create_qvhighlights_subset_manifest.py \
  --annotations data/qvhighlights/highlight_val_release.jsonl \
  --videos-dir data/qvhighlights/videos \
  --limit 20 \
  --output data/qvhighlights/qvhighlights_val_subset_manifest.json
```

Then inspect it and confirm that the number of available videos has increased.
