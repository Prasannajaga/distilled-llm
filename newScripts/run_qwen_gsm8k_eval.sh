#!/usr/bin/env bash
set -euo pipefail

DATASET_PATH="${1:-/media/prasanna/716F26140AED9B67/datasets/GSM8K}"
MAX_SAMPLES="${2:-0}"
SPLIT="${3:-test}"

STAMP="$(date -u +%Y%m%d_%H%M%S)"
OUT_DIR="newoutput/evals/beforeDistillation.${STAMP}"
LOG_PATH="${OUT_DIR}.run.log"

mkdir -p newoutput/evals

echo "[RUN] dataset_path=${DATASET_PATH}"
echo "[RUN] split=${SPLIT}"
echo "[RUN] max_samples=${MAX_SAMPLES}"
echo "[RUN] output_dir=${OUT_DIR}"
echo "[RUN] log=${LOG_PATH}"
echo "[RUN] cwd=$(pwd)"
echo "[RUN] start_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "[RUN] command=uv run python newScripts/eval_qwen_gsm8k.py ..."

PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_HUB_DISABLE_XET=1 HF_HUB_ENABLE_HF_TRANSFER=1 \
uv run python newScripts/eval_qwen_gsm8k.py \
  --dataset-path "${DATASET_PATH}" \
  --split "${SPLIT}" \
  --max-samples "${MAX_SAMPLES}" \
  --max-new-tokens 128 \
  --download-retries 50 \
  --download-retry-wait-sec 20 \
  --debug \
  --log-interval-samples 10 \
  --output-dir "${OUT_DIR}" \
  2>&1 | tee "${LOG_PATH}"
