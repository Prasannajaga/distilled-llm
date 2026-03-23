#!/usr/bin/env bash
set -euo pipefail

DATASET_PATH="${1:-/media/prasanna/716F26140AED9B67/datasets/GSM8K}"
MAX_SAMPLES="${2:-0}"
SPLIT="${3:-test}"
EVAL_NAME="${4:-}"
LOCAL_ONLY="${5:-0}"

EXTRA_ARGS=()
if [[ -n "${EVAL_NAME}" ]]; then
  EXTRA_ARGS+=(--eval-name "${EVAL_NAME}")
fi
if [[ "${LOCAL_ONLY}" == "1" ]]; then
  EXTRA_ARGS+=(--local-models-only)
fi

PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
uv run python newScripts/run_lm_eval_custom.py \
  --dataset-path "${DATASET_PATH}" \
  --max-samples "${MAX_SAMPLES}" \
  --split "${SPLIT}" \
  --device cuda:0 \
  --batch-size 1 \
  --gen-max-toks 128 \
  "${EXTRA_ARGS[@]}"
