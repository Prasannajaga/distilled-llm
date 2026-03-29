# Custom lm-eval GSM8K Report

- Generated (UTC): `2026-03-29T13:41:53.532412+00:00`
- Run ID: `v1`
- Stage: `teacher-sft`
- Experiment: `optimize-loop`
- Parent Eval: `teacher-sft-c2_top1_e1.5_lr3.75e-05_r16_s2048-v1`
- Task: `gsm8k_local_teacher_sft_c2_top1_e2_0_lr3_75e_05_r16_s2048_v1`
- Dataset arrow: `/media/prasanna/716F26140AED9B67/datasets/openai___gsm8k/main/0.0.0/cc7b047b6e5bb11b4f1af84efc572db110a51b3c/gsm8k-test.arrow`
- Dataset jsonl: `newoutput/lm_eval/teacher-sft-c2_top1_e2.0_lr3.75e-05_r16_s2048-v1/gsm8k_local_teacher_sft_c2_top1_e2_0_lr3_75e_05_r16_s2048_v1_test.jsonl`
- Rows: `1319`

## Scores

- Teacher (`output/teacher-opt/c2_top1_e2.0_lr3.75e-05_r16_s2048/merged`): `0.7035633055344959` (metric: `exact_match,marker-priority(local)`)
- Student (`/media/prasanna/716F26140AED9B67/models/models--Qwen--Qwen2-0.5B-Instruct`): `0.3479909021986353` (metric: `exact_match,marker-priority(local)`)
- Teacher lm-eval metric: `0.7028051554207733` (`exact_match,marker-priority`)
- Student lm-eval metric: `0.3411675511751327` (`exact_match,marker-priority`)
- Delta (teacher - student): `0.35557240333586054`

## Sample Head-to-Head (first loaded rows)

- Loaded rows: `1319`
- Teacher wins: `508`
- Student wins: `39`
- Both correct: `420`
- Both wrong: `352`
- Teacher accuracy (loaded rows): `0.7035633055344959`
- Student accuracy (loaded rows): `0.3479909021986353`

## Outputs

- Teacher results: `newoutput/lm_eval/teacher-sft-c2_top1_e2.0_lr3.75e-05_r16_s2048-v1/teacher/results.json`
- Student results: `newoutput/lm_eval/teacher-sft-c2_top1_e2.0_lr3.75e-05_r16_s2048-v1/student/results.json`
- Teacher samples: `newoutput/lm_eval/teacher-sft-c2_top1_e2.0_lr3.75e-05_r16_s2048-v1/teacher/samples.jsonl`
- Student samples: `newoutput/lm_eval/teacher-sft-c2_top1_e2.0_lr3.75e-05_r16_s2048-v1/student/samples.jsonl`
- Comparison: `newoutput/lm_eval/teacher-sft-c2_top1_e2.0_lr3.75e-05_r16_s2048-v1/comparison.json`
- Sample columns: `newoutput/lm_eval/teacher-sft-c2_top1_e2.0_lr3.75e-05_r16_s2048-v1/sample_columns.jsonl`