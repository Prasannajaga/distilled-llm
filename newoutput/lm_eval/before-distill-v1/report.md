# Custom lm-eval GSM8K Report

- Generated (UTC): `2026-03-23T17:24:21.781813+00:00`
- Task: `gsm8k_local_before_distill_v1`
- Dataset arrow: `/media/prasanna/716F26140AED9B67/datasets/openai___gsm8k/main/0.0.0/cc7b047b6e5bb11b4f1af84efc572db110a51b3c/gsm8k-test.arrow`
- Dataset jsonl: `newoutput/lm_eval/before-distill-v1/gsm8k_local_before_distill_v1_test.jsonl`
- Rows: `1319`

## Scores

- Teacher (`/media/prasanna/716F26140AED9B67/models/models--Qwen--Qwen2-Math-1.5B-Instruct`): `0.1607278241091736` (metric: `exact_match,flexible-extract`)
- Student (`/media/prasanna/716F26140AED9B67/models/models--Qwen--Qwen2-0.5B-Instruct`): `0.050037907505686124` (metric: `exact_match,flexible-extract`)
- Delta (teacher - student): `0.11068991660348748`

## Sample Head-to-Head (first loaded rows)

- Loaded rows: `80`
- Teacher wins: `15`
- Student wins: `1`
- Both correct: `0`
- Both wrong: `64`

## Outputs

- Teacher results: `newoutput/lm_eval/before-distill-v1/teacher/results.json`
- Student results: `newoutput/lm_eval/before-distill-v1/student/results.json`
- Teacher samples: `newoutput/lm_eval/before-distill-v1/teacher/samples.jsonl`
- Student samples: `newoutput/lm_eval/before-distill-v1/student/samples.jsonl`
- Comparison: `newoutput/lm_eval/before-distill-v1/comparison.json`