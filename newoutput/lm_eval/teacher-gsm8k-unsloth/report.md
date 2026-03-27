# Custom lm-eval GSM8K Report

- Generated (UTC): `2026-03-26T13:52:16.502147+00:00`
- Task: `gsm8k_local_teacher_gsm8k_unsloth`
- Dataset arrow: `/media/prasanna/716F26140AED9B67/datasets/openai___gsm8k/main/0.0.0/cc7b047b6e5bb11b4f1af84efc572db110a51b3c/gsm8k-test.arrow`
- Dataset jsonl: `newoutput/lm_eval/teacher-gsm8k-unsloth/gsm8k_local_teacher_gsm8k_unsloth_test.jsonl`
- Rows: `1319`

## Scores

- Teacher (`output/teacher-gsm8k-unsloth-merged`): `0.1561789234268385` (metric: `exact_match,flexible-extract`)
- Student (`Qwen/Qwen2-0.5B-Instruct`): `0.050037907505686124` (metric: `exact_match,flexible-extract`)
- Delta (teacher - student): `0.10614101592115238`

## Sample Head-to-Head (first loaded rows)

- Loaded rows: `1319`
- Teacher wins: `171`
- Student wins: `31`
- Both correct: `35`
- Both wrong: `1082`
- Teacher accuracy (loaded rows): `0.1561789234268385`
- Student accuracy (loaded rows): `0.050037907505686124`

## Outputs

- Teacher results: `newoutput/lm_eval/teacher-gsm8k-unsloth/teacher/results.json`
- Student results: `newoutput/lm_eval/teacher-gsm8k-unsloth/student/results.json`
- Teacher samples: `newoutput/lm_eval/teacher-gsm8k-unsloth/teacher/samples.jsonl`
- Student samples: `newoutput/lm_eval/teacher-gsm8k-unsloth/student/samples.jsonl`
- Comparison: `newoutput/lm_eval/teacher-gsm8k-unsloth/comparison.json`
- Sample columns: `newoutput/lm_eval/teacher-gsm8k-unsloth/sample_columns.jsonl`