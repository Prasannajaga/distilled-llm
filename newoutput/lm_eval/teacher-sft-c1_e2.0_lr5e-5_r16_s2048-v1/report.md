# Custom lm-eval GSM8K Report

- Generated (UTC): `2026-03-29T02:06:42.293583+00:00`
- Run ID: `v1`
- Stage: `teacher-sft`
- Experiment: `optimize-loop`
- Parent Eval: `teacher-sft-c1_e1.0_lr1e-4_r16_s2048-v1`
- Task: `gsm8k_local_teacher_sft_c1_e2_0_lr5e_5_r16_s2048_v1`
- Dataset arrow: `/media/prasanna/716F26140AED9B67/datasets/openai___gsm8k/main/0.0.0/cc7b047b6e5bb11b4f1af84efc572db110a51b3c/gsm8k-test.arrow`
- Dataset jsonl: `newoutput/lm_eval/teacher-sft-c1_e2.0_lr5e-5_r16_s2048-v1/gsm8k_local_teacher_sft_c1_e2_0_lr5e_5_r16_s2048_v1_test.jsonl`
- Rows: `1319`

## Scores

- Teacher (`output/teacher-opt/c1_e2.0_lr5e-5_r16_s2048/merged`): `0.7020470053070508` (metric: `exact_match,marker-priority(local)`)
- Student (`/media/prasanna/716F26140AED9B67/models/models--Qwen--Qwen2-0.5B-Instruct`): `0.3479909021986353` (metric: `exact_match,marker-priority(local)`)
- Teacher lm-eval metric: `0.7012888551933283` (`exact_match,marker-priority`)
- Student lm-eval metric: `0.3411675511751327` (`exact_match,marker-priority`)
- Delta (teacher - student): `0.35405610310841545`

## Sample Head-to-Head (first loaded rows)

- Loaded rows: `1319`
- Teacher wins: `504`
- Student wins: `37`
- Both correct: `422`
- Both wrong: `356`
- Teacher accuracy (loaded rows): `0.7020470053070508`
- Student accuracy (loaded rows): `0.3479909021986353`

## Outputs

- Teacher results: `newoutput/lm_eval/teacher-sft-c1_e2.0_lr5e-5_r16_s2048-v1/teacher/results.json`
- Student results: `newoutput/lm_eval/teacher-sft-c1_e2.0_lr5e-5_r16_s2048-v1/student/results.json`
- Teacher samples: `newoutput/lm_eval/teacher-sft-c1_e2.0_lr5e-5_r16_s2048-v1/teacher/samples.jsonl`
- Student samples: `newoutput/lm_eval/teacher-sft-c1_e2.0_lr5e-5_r16_s2048-v1/student/samples.jsonl`
- Comparison: `newoutput/lm_eval/teacher-sft-c1_e2.0_lr5e-5_r16_s2048-v1/comparison.json`
- Sample columns: `newoutput/lm_eval/teacher-sft-c1_e2.0_lr5e-5_r16_s2048-v1/sample_columns.jsonl`