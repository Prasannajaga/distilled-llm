#!/usr/bin/env bash
set -euo pipefail

cd /home/prasanna/coding/distilled-llm

RUN_ID="${RUN_ID:-v2}"
EXP="${EXP:-distill-format-ablation}"
PHASE_NAME="${PHASE_NAME:-next-distill-phase}"
OUTPUT_ROOT="${OUTPUT_ROOT:-newoutput/distill}"
PINNED_SUMMARY="${PINNED_SUMMARY:-newoutput/teacher_optimization_summary_bootstrap.json}"
COMPARISON_JSON="${COMPARISON_JSON:-$(
  PINNED_SUMMARY="$PINNED_SUMMARY" uv run python -c 'import json, os; print(json.loads(open(os.environ["PINNED_SUMMARY"], encoding="utf-8").read())["best_run"]["comparison_json"])'
)}"
PHASE_SLUG="${PHASE_SLUG:-$(
  PHASE_NAME="$PHASE_NAME" RUN_ID="$RUN_ID" uv run python -c 'import os, re; text = f"{os.environ[\"PHASE_NAME\"]}-{os.environ[\"RUN_ID\"]}".strip().lower(); text = re.sub(r"[^a-z0-9._-]+", "-", text); text = re.sub(r"-{2,}", "-", text).strip("-"); print(text or "run")'
)}"
DATASET_MANIFEST_JSON="${DATASET_MANIFEST_JSON:-${OUTPUT_ROOT}/${PHASE_SLUG}/manifests/dataset_manifest.json}"
MAX_FILTERED_ROWS="${MAX_FILTERED_ROWS:-0}"
BENCHMARK_DATASET_PATH="${BENCHMARK_DATASET_PATH:-/media/prasanna/716F26140AED9B67/datasets/GSM8K}"
HELDOUT_DATASET_PATH="${HELDOUT_DATASET_PATH:-}"
FROZEN_EVAL_SIZE="${FROZEN_EVAL_SIZE:-500}"
OVERLAP_THRESHOLD="${OVERLAP_THRESHOLD:-0.0}"
ALLOW_OVERLAP="${ALLOW_OVERLAP:-0}"
DRY_RUN="${DRY_RUN:-0}"

uv run python -m newScripts.build_distill_variants \
  --comparison-json "$COMPARISON_JSON" \
  --output-root "$OUTPUT_ROOT" \
  --phase-name "$PHASE_NAME" \
  --pipeline-run-id "$RUN_ID" \
  --max-filtered-rows "$MAX_FILTERED_ROWS"

DISTILL_ARGS=(
  --dataset-manifest-json "$DATASET_MANIFEST_JSON"
  --summary-json "$PINNED_SUMMARY"
  --experiment-name "$EXP"
  --pipeline-run-id "$RUN_ID"
  --benchmark-dataset-path "$BENCHMARK_DATASET_PATH"
  --frozen-eval-size "$FROZEN_EVAL_SIZE"
  --overlap-threshold "$OVERLAP_THRESHOLD"
)

if [[ -n "$HELDOUT_DATASET_PATH" ]]; then
  DISTILL_ARGS+=(--heldout-dataset-path "$HELDOUT_DATASET_PATH")
fi
if [[ "$DRY_RUN" == "1" ]]; then
  DISTILL_ARGS+=(--dry-run)
fi
if [[ "$ALLOW_OVERLAP" == "1" ]]; then
  DISTILL_ARGS+=(--allow-overlap)
fi

uv run python -m newScripts.run_student_distill_from_teacher "${DISTILL_ARGS[@]}"
uv run python -m newScripts.build_eval_tracking_dashboard
