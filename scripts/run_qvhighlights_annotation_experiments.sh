#!/usr/bin/env bash
set -euo pipefail

ANNOTATIONS="data/qvhighlights/highlight_val_release.jsonl"
OUTPUT_DIR="results/qvhighlights_annotation_rehearsal"

if [[ ! -f "${ANNOTATIONS}" ]]; then
  echo "Missing annotation file: ${ANNOTATIONS}"
  echo "Please add QVHighlights annotations manually. See docs/qvhighlights_data.md"
  exit 1
fi

mkdir -p "${OUTPUT_DIR}"

.venv/bin/python scripts/inspect_annotations.py \
  --annotations "${ANNOTATIONS}" \
  --limit 5

.venv/bin/python scripts/run_frame_score_retrieval.py \
  --annotations "${ANNOTATIONS}" \
  --fps 1 \
  --window-size 4 \
  --stride 2 \
  --aggregation mean \
  --limit 100 \
  --output "${OUTPUT_DIR}/frame_score_limit100_w4_s2_mean.json"

.venv/bin/python scripts/run_frame_score_retrieval.py \
  --annotations "${ANNOTATIONS}" \
  --fps 1 \
  --window-size 4 \
  --stride 2 \
  --aggregation max \
  --limit 100 \
  --output "${OUTPUT_DIR}/frame_score_limit100_w4_s2_max.json"

.venv/bin/python scripts/run_frame_score_retrieval.py \
  --annotations "${ANNOTATIONS}" \
  --fps 1 \
  --window-size 8 \
  --stride 4 \
  --aggregation mean \
  --limit 100 \
  --output "${OUTPUT_DIR}/frame_score_limit100_w8_s4_mean.json"

.venv/bin/python scripts/run_frame_score_retrieval.py \
  --annotations "${ANNOTATIONS}" \
  --fps 1 \
  --window-size 8 \
  --stride 4 \
  --aggregation max \
  --limit 100 \
  --output "${OUTPUT_DIR}/frame_score_limit100_w8_s4_max.json"

.venv/bin/python scripts/run_frame_score_retrieval.py \
  --annotations "${ANNOTATIONS}" \
  --fps 1 \
  --window-size 8 \
  --stride 4 \
  --aggregation mean \
  --smoothing-window 3 \
  --limit 100 \
  --output "${OUTPUT_DIR}/frame_score_limit100_w8_s4_mean_smooth3.json"

.venv/bin/python scripts/collect_results.py \
  --results-dir "${OUTPUT_DIR}" \
  --output "${OUTPUT_DIR}/summary.csv"
