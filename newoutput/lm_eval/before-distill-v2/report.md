# Custom lm-eval GSM8K Report

- Generated (UTC): `2026-03-24T06:10:19.059453+00:00`
- Task: `gsm8k_local_before_distill_v2`
- Dataset arrow: `hf://openai/gsm8k/main`
- Dataset jsonl: `newoutput/lm_eval/before-distill-v2/gsm8k_local_before_distill_v2_test.jsonl`
- Rows: `1319`

## Scores

- Teacher (`/media/prasanna/716F26140AED9B67/models/models--Qwen--Qwen2-Math-1.5B-Instruct`): `0.1607278241091736` (metric: `exact_match,flexible-extract`)
- Student (`/media/prasanna/716F26140AED9B67/models/models--Qwen--Qwen2-0.5B-Instruct`): `0.050037907505686124` (metric: `exact_match,flexible-extract`)
- Delta (teacher - student): `0.11068991660348748`

## Sample Head-to-Head (first loaded rows)

- Loaded rows: `1319`
- Teacher wins: `177`
- Student wins: `31`
- Both correct: `35`
- Both wrong: `1076`
- Teacher accuracy (loaded rows): `0.1607278241091736`
- Student accuracy (loaded rows): `0.050037907505686124`

## Outputs

- Teacher results: `newoutput/lm_eval/before-distill-v2/teacher/results.json`
- Student results: `newoutput/lm_eval/before-distill-v2/student/results.json`
- Teacher samples: `newoutput/lm_eval/before-distill-v2/teacher/samples.jsonl`
- Student samples: `newoutput/lm_eval/before-distill-v2/student/samples.jsonl`
- Comparison: `newoutput/lm_eval/before-distill-v2/comparison.json`
- Sample columns: `newoutput/lm_eval/before-distill-v2/sample_columns.jsonl`