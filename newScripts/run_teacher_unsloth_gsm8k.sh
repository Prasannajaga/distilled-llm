#!/usr/bin/env bash
set -euo pipefail

DATASET_PATH="${1:-/media/prasanna/716F26140AED9B67/datasets/GSM8K}"
BASE_MODEL="${2:-Qwen/Qwen2-Math-1.5B-Instruct}"
MAX_TRAIN="${3:-0}"

OUT_DIR="output/teacher-gsm8k-unsloth-lora"
MERGED_DIR="output/teacher-gsm8k-unsloth-merged"

PYTHONUNBUFFERED=1 uv run python newScripts/train_teacher_unsloth_gsm8k.py \
  --dataset-path "${DATASET_PATH}" \
  --base-model "${BASE_MODEL}" \
  --max-train-samples "${MAX_TRAIN}" \
  --output-dir "${OUT_DIR}" \
  --merged-output-dir "${MERGED_DIR}" \
  --merge-16bit 1
