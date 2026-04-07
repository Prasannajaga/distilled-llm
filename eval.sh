#!/usr/bin/env bash
set -euo pipefail

cd /home/prasanna/coding/distilled-llm

# Source comparison that provides:
# 1) teacher artifacts/scores reuse
# 2) exact dataset_jsonl reuse (same 2k set across all runs)
SOURCE_COMPARISON_JSON="${SOURCE_COMPARISON_JSON:-newoutput/lm_eval/distill-source-train-v1/comparison.json}"

# Run controls
DEVICE="${DEVICE:-cuda:0}"
BATCH_SIZE="${BATCH_SIZE:-1}"
NUM_FEWSHOT="${NUM_FEWSHOT:-8}"
GEN_MAX_TOKS="${GEN_MAX_TOKS:-256}"
EVAL_LIMIT="${EVAL_LIMIT:-}"
USE_EVAL_LIMIT="${USE_EVAL_LIMIT:-0}"
RETRY_COUNT="${RETRY_COUNT:-1}"
GPU_OPTIMIZE_6GB="${GPU_OPTIMIZE_6GB:-1}"
FAIL_ON_PARTIAL="${FAIL_ON_PARTIAL:-1}"

# Logging + grouping
RUN_ID="${RUN_ID:-distilled-train-v1}"
STAGE="${STAGE:-student-distill-eval}"
EXPERIMENT="${EXPERIMENT:-student-format-ablation}"
LOG_ROOT="${LOG_ROOT:-newoutput/new_lm_eval/logs/${RUN_ID}}"
mkdir -p "$LOG_ROOT"
MASTER_LOG="$LOG_ROOT/run.log"

log() {
  printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*" | tee -a "$MASTER_LOG"
}

if [[ -n "${LIMIT:-}" && -z "$EVAL_LIMIT" ]]; then
  log "Detected inherited LIMIT=${LIMIT}, but eval.sh now ignores LIMIT to prevent accidental 1-sample runs."
  log "If you intentionally want a subset run, use USE_EVAL_LIMIT=1 EVAL_LIMIT=<N> (for example USE_EVAL_LIMIT=1 EVAL_LIMIT=100)."
fi
if [[ -n "$EVAL_LIMIT" && "$USE_EVAL_LIMIT" != "1" ]]; then
  log "Detected EVAL_LIMIT=${EVAL_LIMIT}, but USE_EVAL_LIMIT!=1 so limit is ignored."
fi

run_one() {
  local step="$1"
  local eval_name="$2"
  local student_model="$3"
  local out_dir="newoutput/new_lm_eval/${eval_name}"
  local step_log="$LOG_ROOT/${step}-${eval_name}.log"

  log "STEP ${step}/3 start | eval=${eval_name}"

  local -a cmd=(
    uv run python -m newScripts.run_lm_eval_custom
    --student-model "$student_model"
    --reuse-teacher-comparison-json "$SOURCE_COMPARISON_JSON"
    --reuse-dataset-jsonl-from-comparison "$SOURCE_COMPARISON_JSON"
    --eval-name "$eval_name"
    --output-dir "$out_dir"
    --overwrite-output
    --stage "$STAGE"
    --experiment "$EXPERIMENT"
    --run-id "$RUN_ID"
    --device "$DEVICE"
    --batch-size "$BATCH_SIZE"
    --num-fewshot "$NUM_FEWSHOT"
    --gen-max-toks "$GEN_MAX_TOKS"
    --retry-count "$RETRY_COUNT"
    --gpu-optimize-6gb "$GPU_OPTIMIZE_6GB"
    --echo-subprocess 1
    --render-html 1
    --html-limit 200
  )

  if [[ "$USE_EVAL_LIMIT" == "1" && -n "$EVAL_LIMIT" ]]; then
    cmd+=(--limit "$EVAL_LIMIT")
  fi
  log "Limit status | active=$([[ "$USE_EVAL_LIMIT" == "1" && -n "$EVAL_LIMIT" ]] && echo 1 || echo 0) value=${EVAL_LIMIT:-none}"

  {
    printf '[%s] Command: ' "$(date '+%Y-%m-%d %H:%M:%S')"
    printf '%q ' "${cmd[@]}"
    printf '\n'
    "${cmd[@]}"
  } 2>&1 | tee "$step_log"

  local comparison_json="${out_dir}/comparison.json"
  if [[ -f "$comparison_json" ]]; then
    local num_rows loaded_rows
    num_rows="$(jq -r '.num_rows // 0' "$comparison_json" 2>/dev/null || echo 0)"
    loaded_rows="$(jq -r '.sample_comparison.loaded_rows // 0' "$comparison_json" 2>/dev/null || echo 0)"
    if [[ "$num_rows" != "$loaded_rows" ]]; then
      log "STEP ${step}/3 partial-eval detected | eval=${eval_name} | loaded_rows=${loaded_rows} num_rows=${num_rows}"
      if [[ "$FAIL_ON_PARTIAL" == "1" ]]; then
        log "Failing run because FAIL_ON_PARTIAL=1. Set FAIL_ON_PARTIAL=0 to allow subset eval runs."
        exit 2
      fi
    fi
  else
    log "STEP ${step}/3 missing comparison.json | eval=${eval_name} | expected=${comparison_json}"
    exit 2
  fi

  log "STEP ${step}/3 end | eval=${eval_name} | comparison=${out_dir}/comparison.json"
}

if [[ ! -f "$SOURCE_COMPARISON_JSON" ]]; then
  echo "Missing SOURCE_COMPARISON_JSON: $SOURCE_COMPARISON_JSON" >&2
  exit 1
fi

log "3-category eval start"
log "Source comparison: $SOURCE_COMPARISON_JSON"

run_one 1 answer-only-distilled-train-v1 newoutput/distill/next-distill-phase-v1/models/answer_only/merged
run_one 2 short-rationale-distilled-train-v1 newoutput/distill/next-distill-phase-v1/models/short_rationale_only/merged
run_one 3 full-rationale-distilled-train-v1 newoutput/distill/next-distill-phase-v1/models/full_rationale/merged

log "3-category eval complete"
