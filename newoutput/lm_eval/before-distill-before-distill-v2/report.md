# Custom lm-eval GSM8K Report

- Generated (UTC): `2026-03-28T20:16:11.735109+00:00`
- Run ID: `v2`
- Stage: `before-distill`
- Experiment: `before-distill`
- Parent Eval: `None`
- Task: `gsm8k_local_before_distill_before_distill_v2`
- Dataset arrow: `/media/prasanna/716F26140AED9B67/datasets/openai___gsm8k/main/0.0.0/cc7b047b6e5bb11b4f1af84efc572db110a51b3c/gsm8k-test.arrow`
- Dataset jsonl: `newoutput/lm_eval/before-distill-before-distill-v2/gsm8k_local_before_distill_before_distill_v2_test.jsonl`
- Rows: `1319`

## Scores

- Teacher (`/media/prasanna/716F26140AED9B67/models/models--Qwen--Qwen2-Math-1.5B-Instruct`): `0.7437452615617892` (metric: `exact_match,marker-priority(local)`)
- Student (`/media/prasanna/716F26140AED9B67/models/models--Qwen--Qwen2-0.5B-Instruct`): `0.3479909021986353` (metric: `exact_match,marker-priority(local)`)
- Teacher lm-eval metric: `0.7422289613343442` (`exact_match,marker-priority`)
- Student lm-eval metric: `0.3411675511751327` (`exact_match,marker-priority`)
- Delta (teacher - student): `0.3957543593631539`

## Sample Head-to-Head (first loaded rows)

- Loaded rows: `1319`
- Teacher wins: `558`
- Student wins: `36`
- Both correct: `423`
- Both wrong: `302`
- Teacher accuracy (loaded rows): `0.7437452615617892`
- Student accuracy (loaded rows): `0.3479909021986353`

## Outputs

- Teacher results: `newoutput/lm_eval/before-distill-before-distill-v2/teacher/results.json`
- Student results: `newoutput/lm_eval/before-distill-before-distill-v2/student/results.json`
- Teacher samples: `newoutput/lm_eval/before-distill-before-distill-v2/teacher/samples.jsonl`
- Student samples: `newoutput/lm_eval/before-distill-before-distill-v2/student/samples.jsonl`
- Comparison: `newoutput/lm_eval/before-distill-before-distill-v2/comparison.json`
- Sample columns: `newoutput/lm_eval/before-distill-before-distill-v2/sample_columns.jsonl`