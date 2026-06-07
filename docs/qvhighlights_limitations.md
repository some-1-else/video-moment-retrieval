# QVHighlights Limitations

QVHighlights was the initial dataset direction for this Video Moment Retrieval
project because it is a modern benchmark with natural-language queries,
temporal windows, and close connection to the official Moment-DETR release.

The final coursework experiments were moved to Charades-STA because
QVHighlights raw-video availability made a reproducible local benchmark
difficult within the project scope.

## Why QVHighlights Was Considered

QVHighlights provides JSONL annotations with fields such as:

```text
qid
query
duration
vid
relevant_clip_ids
saliency_scores
relevant_windows
```

The local loader maps these fields into the project `MomentRetrievalSample`
format. Annotation-only checks confirmed that validation annotations could be
read and converted into the same retrieval/evaluation interface used elsewhere
in the project.

## What Worked

The following exploratory pieces worked locally:

- QVHighlights annotation parsing;
- small annotation-only rehearsal runs;
- subset manifest creation for selected validation samples;
- parsing QVHighlights `video_id` values into YouTube id and clip timestamps;
- limited CLIP retrieval on videos that were already available and readable
  locally;
- embedding cache checks on a tiny readable subset.

These experiments are preserved as research history in ignored
`.agent_memory/` paths. They are not part of the public final results.

## What Made QVHighlights Hard to Use

QVHighlights videos are tied to external web-hosted sources. Local
reproducibility depends on whether the corresponding YouTube videos are still
available, downloadable, region-accessible, and decodable by the local
OpenCV/FFmpeg setup.

Observed limitations:

- many selected videos were missing locally;
- some source videos were unavailable or blocked;
- some downloaded files existed but were unreadable by OpenCV;
- full-dataset video download would be large, slow, and brittle;
- official Moment-DETR dataset-level inference expects pre-extracted
  QVHighlights features rather than arbitrary local raw videos.

Because of this, small QVHighlights experiments could check the pipeline but
could not support a reliable public benchmark claim.

## Preliminary Results Are Not the Main Benchmark

Exploratory QVHighlights runs used tiny locally available subsets, including
2-video and 9-readable-video settings. These runs were useful for validating
data loading, video availability checks, CLIP scoring, and embedding cache
behavior, but they are not representative of the full QVHighlights validation
split.

Therefore:

- QVHighlights numbers should not be reported as final benchmark performance;
- QVHighlights outputs are kept as local research history;
- the public `results/` folder intentionally keeps only final coursework
  Charades-STA results.

## Why Final Experiments Use Charades-STA

Charades-STA was chosen for the final coursework line because it allowed a
clearer local reproduction path:

- annotation format is simple text;
- local videos can be prepared from the Charades archive;
- videos are short enough for CPU CLIP experiments;
- fixed subsets can be verified and reused across sweeps;
- CLIP and Moment-DETR probe results can be evaluated with the same temporal
  localization metrics.

This switch keeps the project focused on reproducible Video Moment Retrieval
experiments rather than on brittle external-video acquisition.

## Current Status

QVHighlights support remains in `src/data/qvhighlights.py` because it is useful
for preserving the general moment-retrieval sample format and for future work.
However, QVHighlights is not the final benchmark path for this coursework.
