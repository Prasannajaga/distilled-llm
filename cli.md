# CLI Usage Guide

This file documents command-line usage for scripts in `newScripts/`.

## Quick Start: Custom Eval (Student Base vs Teacher Instruct)

Run this first to compare base student against instruct teacher on GSM8K test split:

```bash
uv run python newScripts/run_lm_eval_custom.py \
  --dataset-path /media/prasanna/716F26140AED9B67/datasets/GSM8K \
  --split test \
  --teacher-model /media/prasanna/716F26140AED9B67/models/models--Qwen--Qwen2-Math-1.5B-Instruct \
  --student-model /media/prasanna/716F26140AED9B67/models/models--Qwen--Qwen2-0.5B \
  --device cuda:0 \
  --batch-size 1 \
  --num-fewshot 8 \
  --gen-max-toks 256 \
  --fewshot-cot 1 \
  --eval-name before-distill-teacher-vs-student-base-v1 \
  --stage before-distill \
  --experiment teacher-opt \
  --run-id v1 \
  --overwrite-output \
  --max-samples 500
```

Notes:

- If your local model path differs, replace `--teacher-model` / `--student-model` paths.
- Outputs are written under `newoutput/lm_eval/<eval-name>/`.

---

## `run_lm_eval_custom.py`

Purpose:

- Runs lm-eval on teacher + student, computes comparison metrics, writes `comparison.json`, `report.md`, and `report.html`.

Usage:

```bash
uv run python newScripts/run_lm_eval_custom.py [options]
```

Key options:

- Data/control: `--dataset-path`, `--split`, `--max-samples`, `--limit`
- Models: `--teacher-model`, `--student-model`, `--local-models-only`
- Generation/eval: `--num-fewshot`, `--gen-max-toks`, `--fewshot-cot`, `--batch-size`, `--device`
- Tracking: `--eval-name`, `--stage`, `--experiment`, `--run-id`, `--parent-eval`, `--notes`
- Reliability: `--retry-count`, `--gpu-optimize-6gb`, fallback device/max-token flags
- Reuse student eval: `--reuse-student-comparison-json <comparison.json>`
- Output/reporting: `--overwrite-output`, `--render-html`, `--html-limit`, `--html-title`

Reuse-student example (teacher-only eval runtime savings):

```bash
uv run python newScripts/run_lm_eval_custom.py \
  --dataset-path /media/prasanna/716F26140AED9B67/datasets/GSM8K \
  --split test \
  --teacher-model /path/to/new/teacher/merged \
  --student-model /media/prasanna/716F26140AED9B67/models/models--Qwen--Qwen2-0.5B \
  --reuse-student-comparison-json newoutput/lm_eval/before-distill-teacher-opt-v1/comparison.json \
  --eval-name teacher-sft-c1-v1 \
  --stage teacher-sft \
  --experiment teacher-opt \
  --run-id v1 \
  --overwrite-output
```

---

## `run_before_distill_eval.py`

Purpose:

- Runs the baseline before any distillation and creates a stable baseline `comparison.json`.

Usage:

```bash
uv run python newScripts/run_before_distill_eval.py [options]
```

Key options:

- `--dataset-path`, `--split`, `--max-samples`
- `--teacher-model`, `--student-model`
- `--device`, `--batch-size`, `--num-fewshot`, `--gen-max-toks`, `--limit`
- `--experiment-name`, `--pipeline-run-id`, `--eval-name`, `--overwrite-output`, `--dry-run`

Example:

```bash
uv run python newScripts/run_before_distill_eval.py \
  --experiment-name teacher-opt \
  --pipeline-run-id v1 \
  --overwrite-output
```

---

## `run_teacher_optimization_loop.py`

Purpose:

- Coarse-to-fine teacher SFT/refinement loop with gating and summary output.

Usage:

```bash
uv run python newScripts/run_teacher_optimization_loop.py [options]
```

Key options:

- Data/eval: `--dataset-path`, `--split`, `--max-samples`, `--limit`
- Models: `--base-model`, `--student-model`
- Teacher train params: `--train-batch-size`, `--train-grad-accum`, `--load-in-4bit`, `--lora-r`, `--lora-alpha`, `--lora-dropout`
- Search space: `--seed-learning-rate`, `--seed-epochs`, `--seed-max-seq-length`, `--max-candidates-per-cycle`
- Loop control: `--max-cycles`, `--hard-slice-rows`, `--oom-retries`, `--min-train-seq-length`
- Milestone gate: `--teacher-em-target`, `--plateau-delta`, `--plateau-cycles`
- Tracking/outputs: `--experiment-name`, `--pipeline-run-id`, `--baseline-comparison-json`, `--work-root`, `--summary-out`, `--dry-run`

Example:

```bash
uv run python newScripts/run_teacher_optimization_loop.py \
  --experiment-name teacher-opt \
  --pipeline-run-id v1 \
  --baseline-comparison-json newoutput/lm_eval/before-distill-teacher-opt-v1/comparison.json \
  --max-cycles 2
```

---

## `train_teacher_unsloth_gsm8k.py`

Purpose:

- Unsloth LoRA fine-tune for teacher on GSM8K-style data.

Usage:

```bash
uv run python newScripts/train_teacher_unsloth_gsm8k.py [options]
```

Key options:

- Data: `--dataset-path`, `--split`, `--train-jsonl`, `--eval-jsonl`, `--max-train-samples`, `--max-eval-samples`
- Model: `--base-model`, `--max-seq-length`, `--load-in-4bit`
- LoRA: `--lora-r`, `--lora-alpha`, `--lora-dropout`
- Train: `--batch-size`, `--grad-accum`, `--epochs`, `--learning-rate`, `--warmup-ratio`, `--weight-decay`
- Output/cache: `--output-dir`, `--merged-output-dir`, `--merge-16bit`, `--dataset-cache-root`, `--use-dataset-cache`
- Formatting: `--answer-style`

Example:

```bash
uv run python newScripts/train_teacher_unsloth_gsm8k.py \
  --split train \
  --epochs 1.0 \
  --learning-rate 2e-5 \
  --output-dir newoutput/teacher-lora-c1 \
  --merged-output-dir newoutput/teacher-merged-c1
```

---

## `train_teacher_full_sft_gsm8k.py`

Purpose:

- Full SFT (no LoRA) teacher training.

Usage:

```bash
uv run python newScripts/train_teacher_full_sft_gsm8k.py [options]
```

Key options:

- Data: `--dataset-path`, `--split`, `--train-jsonl`, `--eval-jsonl`, `--max-train-samples`, `--max-eval-samples`
- Train: `--batch-size`, `--eval-batch-size`, `--grad-accum`, `--epochs`, `--learning-rate`, `--warmup-ratio`, `--weight-decay`
- Runtime/output: `--max-seq-length`, `--gradient-checkpointing`, `--output-dir`, `--merged-output-dir`, `--merge-16bit`
- Formatting: `--answer-style`

---

## `build_distill_gsm8k.py`

Purpose:

- Converts `comparison.json` into trainable JSONL slices for distillation/refinement.

Usage:

```bash
uv run python newScripts/build_distill_gsm8k.py \
  --comparison-json <path> \
  --output-jsonl <path> \
  --mode <distill_high_conf|distill_strict|hard_failures|all_gold> [options]
```

Key options:

- `--max-rows` (<=0 means all)
- `--max-answer-chars`

Examples:

```bash
uv run python newScripts/build_distill_gsm8k.py \
  --comparison-json newoutput/lm_eval/teacher-sft-c1-v1/comparison.json \
  --output-jsonl newoutput/distill/student_distill_data.jsonl \
  --mode distill_strict
```

```bash
uv run python newScripts/build_distill_gsm8k.py \
  --comparison-json newoutput/lm_eval/teacher-sft-c1-v1/comparison.json \
  --output-jsonl output/teacher-opt/hard_slice_cycle_3.jsonl \
  --mode hard_failures \
  --max-rows 1024
```

---

## `train_student_unsloth_distill.py`

Purpose:

- Trains student from distillation JSONL via Unsloth LoRA and optionally exports merged model.

Usage:

```bash
uv run python newScripts/train_student_unsloth_distill.py \
  --train-jsonl <path> [options]
```

Key options:

- Required: `--train-jsonl`
- Data: `--eval-jsonl`, `--max-train-samples`, `--max-eval-samples`
- Model/LoRA: `--base-model`, `--max-seq-length`, `--load-in-4bit`, `--lora-r`, `--lora-alpha`, `--lora-dropout`
- Train: `--batch-size`, `--grad-accum`, `--epochs`, `--learning-rate`, `--warmup-ratio`, `--weight-decay`
- Output: `--output-dir`, `--merged-output-dir`, `--merge-16bit`

---

## `build_distill_variants.py`

Purpose:

- First step of the new distillation flow.
- Reads a source `comparison.json`, filters `teacher correct / student wrong` rows, builds the three reusable training variants, and writes a dataset manifest JSON.

Usage:

```bash
uv run python -m newScripts.build_distill_variants \
  --comparison-json <comparison.json> [options]
```

Key options:

- Required: `--comparison-json`
- Output/layout: `--output-root`, `--phase-name`, `--pipeline-run-id`
- Filtering/rendering: `--max-filtered-rows`, `--max-answer-chars`, `--short-rationale-max-sentences`, `--short-rationale-max-chars`

Example:

```bash
uv run python -m newScripts.build_distill_variants \
  --comparison-json newoutput/lm_eval/before-distill-before-distill-v2/comparison.json \
  --output-root newoutput/distill \
  --phase-name next-distill-phase \
  --pipeline-run-id v2 \
  --max-filtered-rows 5000
```

Outputs:

- `newoutput/distill/<phase-slug>/datasets/answer_only.jsonl`
- `newoutput/distill/<phase-slug>/datasets/short_rationale.jsonl`
- `newoutput/distill/<phase-slug>/datasets/full_rationale.jsonl`
- `newoutput/distill/<phase-slug>/manifests/dataset_manifest.json`
- `newoutput/distill/<phase-slug>/manifests/filtered_subset_summary.json`

---

## `build_distill_mix.py`

Purpose:

- Rebuilds the winner-format `gold`, `pure`, and `mixed` datasets from a previously generated dataset manifest.

Usage:

```bash
uv run python -m newScripts.build_distill_mix \
  --dataset-manifest-json <dataset_manifest.json> \
  --winner-format <answer_only|short_rationale|full_rationale> [options]
```

Key options:

- Required: `--dataset-manifest-json`, `--winner-format`
- Mixing: `--mixed-gold-ratio`, `--mixed-seed`

Example:

```bash
uv run python -m newScripts.build_distill_mix \
  --dataset-manifest-json newoutput/distill/next-distill-phase-v2/manifests/dataset_manifest.json \
  --winner-format short_rationale \
  --mixed-gold-ratio 0.30 \
  --mixed-seed 42
```

Outputs:

- `newoutput/distill/<phase-slug>/datasets/<winner-format>_pure.jsonl`
- `newoutput/distill/<phase-slug>/datasets/<winner-format>_gold.jsonl`
- `newoutput/distill/<phase-slug>/datasets/<winner-format>_mixed.jsonl`
- `newoutput/distill/<phase-slug>/manifests/mix_manifest_<winner-format>.json`

---

## `run_student_distill_from_teacher.py`

Purpose:

- Thin orchestration layer for the new distillation pipeline.
- Consumes a dataset manifest, trains the three format variants, evaluates benchmark + heldout sets, selects the winner, builds pure/mixed datasets, and writes the final phase summary.

Usage:

```bash
uv run python -m newScripts.run_student_distill_from_teacher \
  --dataset-manifest-json <dataset_manifest.json> [options]
```

Key options:

- Required: `--dataset-manifest-json`
- Teacher/run context: `--summary-json`, `--teacher-model`, `--experiment-name`, `--pipeline-run-id`, `--parent-eval`
- Student train: `--student-base-model`, `--student-epochs`, `--student-lr`, `--student-batch-size`, `--student-grad-accum`, `--student-max-seq-length`, `--student-max-train-samples`, `--student-max-eval-samples`, `--student-seed`, `--train-val-ratio`, `--split-seed`
- Benchmark/heldout eval: `--benchmark-dataset-path`, `--benchmark-split`, `--benchmark-max-samples`, `--heldout-dataset-path`, `--heldout-split`, `--heldout-max-samples`
- Generation/runtime: `--num-fewshot`, `--gen-max-toks`, `--device`, `--batch-size`, `--limit`
- Mixing/control: `--mixed-gold-ratio`, `--mixed-seed`, `--dry-run`

Example:

```bash
uv run python -m newScripts.run_student_distill_from_teacher \
  --dataset-manifest-json newoutput/distill/next-distill-phase-v2/manifests/dataset_manifest.json \
  --summary-json newoutput/teacher_optimization_summary_bootstrap.json \
  --experiment-name distill-format-ablation \
  --pipeline-run-id v2
```

Note:

- This script no longer builds the initial variant datasets inline. Run `build_distill_variants.py` first.

---

## `run.sh`

Purpose:

- Canonical full pipeline wrapper for the new distillation flow.
- Resolves the source `comparison.json` from the teacher summary, builds variant datasets, runs the distillation orchestrator, and rebuilds the eval dashboard.

Usage:

```bash
./newScripts/run.sh
```

Environment variables:

- `RUN_ID`
- `EXP`
- `PHASE_NAME`
- `OUTPUT_ROOT`
- `PINNED_SUMMARY`
- `COMPARISON_JSON`
- `PHASE_SLUG`
- `DATASET_MANIFEST_JSON`
- `MAX_FILTERED_ROWS`
- `BENCHMARK_DATASET_PATH`
- `HELDOUT_DATASET_PATH`
- `DRY_RUN`

Example:

```bash
RUN_ID=v2 \
PHASE_NAME=next-distill-phase \
MAX_FILTERED_ROWS=5000 \
./newScripts/run.sh
```

Dry-run example:

```bash
RUN_ID=smoke1 \
PHASE_NAME=distill-smoke \
MAX_FILTERED_ROWS=5 \
DRY_RUN=1 \
./newScripts/run.sh
```

---

## `check_distill_gate.py`

Purpose:

- CI/guardrail utility. Exits with non-zero code when gate is not passed.

Usage:

```bash
uv run python newScripts/check_distill_gate.py \
  --summary-json newoutput/teacher_optimization_summary.json
```

---

## `build_eval_tracking_dashboard.py`

Purpose:

- Scans all eval runs and writes dashboard summaries (`json`, `md`, `html`, `csv`).

Usage:

```bash
uv run python newScripts/build_eval_tracking_dashboard.py [options]
```

Key options:

- `--lm-eval-root`
- `--out-json`, `--out-md`, `--out-html`, `--out-csv`

Example:

```bash
uv run python newScripts/build_eval_tracking_dashboard.py \
  --lm-eval-root newoutput/lm_eval
```

---

## `render_lm_eval_html.py`

Purpose:

- Helper module imported by `run_lm_eval_custom.py` to render `report.html`.

CLI:

- No standalone CLI entrypoint in current version.
