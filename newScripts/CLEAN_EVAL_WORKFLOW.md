# Clean Distillation Eval Workflow

## 1) Build format datasets (teacher-correct/student-wrong base pool)
```bash
uv run python -m newScripts.build_distill_variants \
  --comparison-json newoutput/lm_eval/distill-source-train-v1/comparison.json \
  --output-root newoutput/distill \
  --phase-name next-distill-phase \
  --pipeline-run-id clean-v1
```

## 2) Run distillation pipeline with frozen clean eval + overlap guardrails
```bash
uv run python -m newScripts.run_student_distill_from_teacher \
  --dataset-manifest-json newoutput/distill/next-distill-phase-clean-v1/manifests/dataset_manifest.json \
  --summary-json newoutput/teacher_optimization_summary_bootstrap.json \
  --experiment-name distill-format-clean \
  --pipeline-run-id clean-v1 \
  --benchmark-dataset-path /media/prasanna/716F26140AED9B67/datasets/GSM8K \
  --benchmark-split test \
  --benchmark-max-samples 500 \
  --heldout-dataset-path /media/prasanna/716F26140AED9B67/datasets/GSM8K \
  --heldout-split train \
  --heldout-max-samples 500 \
  --frozen-eval-source benchmark \
  --frozen-eval-size 500 \
  --overlap-threshold 0.0
```

## 3) Build mixed-bucket recipe datasets directly (pure / mix_b / mix_c)
```bash
uv run python -m newScripts.build_distill_mix \
  --dataset-manifest-json newoutput/distill/next-distill-phase-clean-v1/manifests/dataset_manifest.json \
  --winner-format answer_only \
  --recipe-seed 42
```

## 4) Clean eval for one checkpoint (explicit overlap refs + no silent limit)
```bash
uv run python -m newScripts.run_lm_eval_custom \
  --dataset-jsonl newoutput/distill/next-distill-phase-clean-v1/datasets/frozen_eval_clean.jsonl \
  --teacher-model /path/to/teacher \
  --student-model /path/to/student \
  --eval-name clean-final-student \
  --overlap-reference distill_train=/path/to/train.jsonl \
  --overlap-reference distill_val=/path/to/val.jsonl \
  --overlap-threshold 0.0
```

## 5) Optional explicit subset eval (must be opt-in)
```bash
uv run python -m newScripts.run_lm_eval_custom \
  --dataset-jsonl newoutput/distill/next-distill-phase-clean-v1/datasets/frozen_eval_clean.jsonl \
  --teacher-model /path/to/teacher \
  --student-model /path/to/student \
  --limit 100 \
  --allow-partial-eval
```
