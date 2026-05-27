#!/usr/bin/env bash
set -euo pipefail

ANNOTATIONS="data/sample/qvhighlights_sample.jsonl"
OUTPUT_DIR="results/sample_experiments"

mkdir -p "${OUTPUT_DIR}"

.venv/bin/python scripts/run_frame_score_retrieval.py \
  --annotations "${ANNOTATIONS}" \
  --fps 1 \
  --window-size 4 \
  --stride 2 \
  --aggregation mean \
  --output "${OUTPUT_DIR}/frame_score_w4_s2_mean.json"

.venv/bin/python scripts/run_frame_score_retrieval.py \
  --annotations "${ANNOTATIONS}" \
  --fps 1 \
  --window-size 4 \
  --stride 2 \
  --aggregation max \
  --output "${OUTPUT_DIR}/frame_score_w4_s2_max.json"

.venv/bin/python scripts/run_frame_score_retrieval.py \
  --annotations "${ANNOTATIONS}" \
  --fps 1 \
  --window-size 8 \
  --stride 4 \
  --aggregation mean \
  --output "${OUTPUT_DIR}/frame_score_w8_s4_mean.json"

.venv/bin/python scripts/run_frame_score_retrieval.py \
  --annotations "${ANNOTATIONS}" \
  --fps 1 \
  --window-size 8 \
  --stride 4 \
  --aggregation max \
  --output "${OUTPUT_DIR}/frame_score_w8_s4_max.json"

.venv/bin/python scripts/run_frame_score_retrieval.py \
  --annotations "${ANNOTATIONS}" \
  --fps 1 \
  --window-size 4 \
  --stride 2 \
  --aggregation mean \
  --smoothing-window 3 \
  --output "${OUTPUT_DIR}/frame_score_w4_s2_mean_smooth3.json"

.venv/bin/python scripts/collect_results.py \
  --results-dir "${OUTPUT_DIR}" \
  --output "${OUTPUT_DIR}/summary.csv"
