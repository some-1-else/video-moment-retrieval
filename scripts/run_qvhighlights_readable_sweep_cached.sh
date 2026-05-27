#!/usr/bin/env bash
set -euo pipefail

MANIFEST="data/qvhighlights/qvhighlights_val_diverse50_subset_manifest.json"
VALIDATION="data/qvhighlights/qvhighlights_val_diverse50_video_validation.json"
RESULTS_DIR="results/qvhighlights_readable_sweep_cached"
EMBEDDINGS_CACHE_DIR="results/embeddings/qvhighlights_readable"
FPS="1"
MODEL_NAME="ViT-B/32"

mkdir -p "${RESULTS_DIR}"

.venv/bin/python scripts/run_qvhighlights_available_clip_retrieval.py \
  --manifest "${MANIFEST}" \
  --validation-report "${VALIDATION}" \
  --fps "${FPS}" \
  --window-size 4 \
  --stride 2 \
  --aggregation mean \
  --model-name "${MODEL_NAME}" \
  --use-cache \
  --embeddings-cache-dir "${EMBEDDINGS_CACHE_DIR}" \
  --output "${RESULTS_DIR}/clip_w4_s2_mean.json"

.venv/bin/python scripts/run_qvhighlights_available_clip_retrieval.py \
  --manifest "${MANIFEST}" \
  --validation-report "${VALIDATION}" \
  --fps "${FPS}" \
  --window-size 4 \
  --stride 2 \
  --aggregation max \
  --model-name "${MODEL_NAME}" \
  --use-cache \
  --embeddings-cache-dir "${EMBEDDINGS_CACHE_DIR}" \
  --output "${RESULTS_DIR}/clip_w4_s2_max.json"

.venv/bin/python scripts/run_qvhighlights_available_clip_retrieval.py \
  --manifest "${MANIFEST}" \
  --validation-report "${VALIDATION}" \
  --fps "${FPS}" \
  --window-size 8 \
  --stride 4 \
  --aggregation mean \
  --model-name "${MODEL_NAME}" \
  --use-cache \
  --embeddings-cache-dir "${EMBEDDINGS_CACHE_DIR}" \
  --output "${RESULTS_DIR}/clip_w8_s4_mean.json"

.venv/bin/python scripts/run_qvhighlights_available_clip_retrieval.py \
  --manifest "${MANIFEST}" \
  --validation-report "${VALIDATION}" \
  --fps "${FPS}" \
  --window-size 8 \
  --stride 4 \
  --aggregation max \
  --model-name "${MODEL_NAME}" \
  --use-cache \
  --embeddings-cache-dir "${EMBEDDINGS_CACHE_DIR}" \
  --output "${RESULTS_DIR}/clip_w8_s4_max.json"

.venv/bin/python scripts/run_qvhighlights_available_clip_retrieval.py \
  --manifest "${MANIFEST}" \
  --validation-report "${VALIDATION}" \
  --fps "${FPS}" \
  --window-size 16 \
  --stride 8 \
  --aggregation mean \
  --model-name "${MODEL_NAME}" \
  --use-cache \
  --embeddings-cache-dir "${EMBEDDINGS_CACHE_DIR}" \
  --output "${RESULTS_DIR}/clip_w16_s8_mean.json"

.venv/bin/python scripts/run_qvhighlights_available_clip_retrieval.py \
  --manifest "${MANIFEST}" \
  --validation-report "${VALIDATION}" \
  --fps "${FPS}" \
  --window-size 16 \
  --stride 8 \
  --aggregation max \
  --model-name "${MODEL_NAME}" \
  --use-cache \
  --embeddings-cache-dir "${EMBEDDINGS_CACHE_DIR}" \
  --output "${RESULTS_DIR}/clip_w16_s8_max.json"

.venv/bin/python scripts/run_qvhighlights_available_clip_retrieval.py \
  --manifest "${MANIFEST}" \
  --validation-report "${VALIDATION}" \
  --fps "${FPS}" \
  --window-size 32 \
  --stride 16 \
  --aggregation mean \
  --model-name "${MODEL_NAME}" \
  --use-cache \
  --embeddings-cache-dir "${EMBEDDINGS_CACHE_DIR}" \
  --output "${RESULTS_DIR}/clip_w32_s16_mean.json"

.venv/bin/python scripts/run_qvhighlights_available_clip_retrieval.py \
  --manifest "${MANIFEST}" \
  --validation-report "${VALIDATION}" \
  --fps "${FPS}" \
  --window-size 32 \
  --stride 16 \
  --aggregation max \
  --model-name "${MODEL_NAME}" \
  --use-cache \
  --embeddings-cache-dir "${EMBEDDINGS_CACHE_DIR}" \
  --output "${RESULTS_DIR}/clip_w32_s16_max.json"

.venv/bin/python scripts/collect_results.py \
  --results-dir "${RESULTS_DIR}" \
  --output "${RESULTS_DIR}/summary.csv"
