# Video Frame Extraction

Frame extraction is the bridge between local video files and the planned CLIP-based baseline. In the future baseline, frames will be sampled from a video at fixed timestamps, encoded with a CLIP image encoder, and compared with the text query embedding.

## Current Scope

The current implementation only supports extraction from a local video file provided by the user. It does not download QVHighlights videos, does not process the full dataset, and does not run CLIP.

This layer is intended as a small smoke-testable component:

```text
local video file -> sampled timestamps -> decoded frames
```

The existing `src/video/frame_sampling.py` module generates timestamps from `duration` and `fps`. The new frame extraction module uses those timestamps to read frames from a local video.

## Dependency

The implementation uses `opencv-python` for video reading:

```text
opencv-python
```

OpenCV is imported lazily inside the frame extraction functions, so most tests and non-video workflows do not require video decoding to run.

## Optional Local Smoke Test

When a local video file is available, run:

```bash
.venv/bin/python scripts/inspect_video.py \
  --video path/to/video.mp4 \
  --fps 1
```

The script prints:

- estimated duration;
- number of sampled timestamps;
- number of extracted frames;
- first frame shape, if at least one frame was extracted.

## Timestamp Behavior

`extract_frames_at_timestamps(video_path, timestamps)` expects timestamps in seconds. Negative timestamps are rejected. Timestamps greater than or equal to the estimated video duration are rejected with `ValueError`.

This is intentionally strict: silently skipping out-of-range timestamps can hide bugs in sampling logic.

## Limitations

- QVHighlights videos are not downloaded by this project step.
- Dataset-level video processing is not implemented yet.
- Different video codecs may behave differently depending on local OpenCV/FFmpeg support.
- Some systems may need additional FFmpeg support for certain video formats.
- Large videos can be slow to seek frame-by-frame.
- This module only prepares frames; it does not compute CLIP embeddings.
