# Eval Tracking Dashboard

- Total eval runs scanned: `33`
- Experiments: `8`
- Teacher optimization runs: `4`

## Teacher Optimization Best

- Best eval: `teacher-sft-c2_top1_e1.5_lr3.75e-05_r16_s2048-v1`
- Best teacher EM: `0.721759`
- Student EM on same eval: `0.347991`
- Delta (teacher - student): `0.373768`
- Experiment / Run ID: `optimize-loop` / `v1`

## Teacher Optimization Ranking

| Rank | Eval Name | Stage | Teacher EM | Student EM | Delta | Cycle | Epochs | LR | Seq | Created UTC |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | teacher-sft-c2_top1_e1.5_lr3.75e-05_r16_s2048-v1 | teacher-sft | 0.721759 | 0.347991 | 0.373768 | 2 | 1.5 | 3.75e-05 | 2048 | 2026-03-29T10:38:40.732908+00:00 |
| 2 | teacher-sft-c2_top1_e2.0_lr3.75e-05_r16_s2048-v1 | teacher-sft | 0.703563 | 0.347991 | 0.355572 | 2 | 2.0 | 3.75e-05 | 2048 | 2026-03-29T13:41:53.532412+00:00 |
| 3 | teacher-sft-c1_e2.0_lr5e-5_r16_s2048-v1 | teacher-sft | 0.702047 | 0.347991 | 0.354056 | 1 | 2.0 | 5e-05 | 2048 | 2026-03-29T02:06:42.293583+00:00 |
| 4 | teacher-sft-c1_seed_e1.5_lr7.50e-05_s2048_r16_a16_d0.00-qwen-template-fix-v1 | teacher-sft | 0.338135 | 0.347991 | -0.009856 | 1 | 1.5 | 7.5e-05 | 2048 | 2026-03-31T12:12:54.435566+00:00 |

## Experiment Timeline

| Experiment | Run ID | Stage | Eval Name | Teacher EM | Student EM | Created UTC |
|---|---|---|---|---:|---:|---|
| before-distill | v2 | before-distill | before-distill-before-distill-v2 | 0.743745 | 0.347991 | 2026-03-28T20:16:11.735109+00:00 |
| default | adhoc-20260407-023807 | adhoc | clean-final-mix-b-confirm | 0.770000 | 0.416000 | 2026-04-07T03:38:26.790284+00:00 |
| distill-format-ablation | v1 | distill-source | distill-source-train-v1 | 0.817500 | 0.660000 | 2026-04-01T19:12:18.180051+00:00 |
| distill-format-ablation | v2 | distill-format-benchmark | distill-benchmark-next-distill-phase-v2-format-answer_only-v2 | 0.740000 | 0.402000 | 2026-04-01T05:00:29.351298+00:00 |
| distill-format-ablation | v2 | distill-format-benchmark | distill-benchmark-next-distill-phase-v2-format-full_rationale-v2 | 0.740000 | 0.398000 | 2026-04-01T08:53:12.083319+00:00 |
| distill-format-ablation | v2 | distill-format-benchmark | distill-benchmark-next-distill-phase-v2-format-short_rationale-v2 | 0.740000 | 0.394000 | 2026-04-01T06:52:05.934242+00:00 |
| distill-format-ablation | v2 | distill-format-heldout | distill-heldout-next-distill-phase-v2-format-answer_only-v2 | 0.806000 | 0.638000 | 2026-04-01T05:55:33.201451+00:00 |
| distill-format-ablation | v2 | distill-format-heldout | distill-heldout-next-distill-phase-v2-format-full_rationale-v2 | 0.806000 | 0.644000 | 2026-04-01T09:49:17.443117+00:00 |
| distill-format-ablation | v2 | distill-format-heldout | distill-heldout-next-distill-phase-v2-format-short_rationale-v2 | 0.806000 | 0.640000 | 2026-04-01T07:50:40.210507+00:00 |
| distill-format-clean | clean-v1 | distill-format-benchmark | distill-benchmark-next-distill-phase-clean-v1-format-answer_only-clean-v1 | 0.770000 | 0.402000 | 2026-04-06T04:56:03.032954+00:00 |
| distill-format-clean | clean-v1 | distill-format-benchmark | distill-benchmark-next-distill-phase-clean-v1-format-full_rationale-clean-v1 | 0.770000 | 0.408000 | 2026-04-06T10:49:09.607709+00:00 |
| distill-format-clean | clean-v1 | distill-format-benchmark | distill-benchmark-next-distill-phase-clean-v1-format-short_rationale-clean-v1 | 0.770000 | 0.392000 | 2026-04-06T07:57:49.036437+00:00 |
| distill-format-clean | clean-v1 | distill-format-benchmark | distill-benchmark-next-distill-phase-clean-v1-winner-full_rationale-recipe_mix_b-clean-v1 | 0.770000 | 0.416000 | 2026-04-06T16:44:21.469532+00:00 |
| distill-format-clean | clean-v1 | distill-format-benchmark | distill-benchmark-next-distill-phase-clean-v1-winner-full_rationale-recipe_mix_c-clean-v1 | 0.770000 | 0.412000 | 2026-04-06T19:39:22.040072+00:00 |
| distill-format-clean | clean-v1 | distill-format-benchmark | distill-benchmark-next-distill-phase-clean-v1-winner-full_rationale-recipe_pure_target-clean-v1 | 0.770000 | 0.408000 | 2026-04-06T13:46:33.310967+00:00 |
| distill-format-clean | clean-v1 | distill-format-final | distill-final-next-distill-phase-clean-v1-format-answer_only-clean-v1 | 0.770000 | 0.402000 | 2026-04-06T06:50:30.883184+00:00 |
| distill-format-clean | clean-v1 | distill-format-final | distill-final-next-distill-phase-clean-v1-format-full_rationale-clean-v1 | 0.770000 | 0.408000 | 2026-04-06T12:45:53.305613+00:00 |
| distill-format-clean | clean-v1 | distill-format-final | distill-final-next-distill-phase-clean-v1-format-short_rationale-clean-v1 | 0.770000 | 0.392000 | 2026-04-06T09:50:55.690392+00:00 |
| distill-format-clean | clean-v1 | distill-format-final | distill-final-next-distill-phase-clean-v1-winner-full_rationale-recipe_mix_b-clean-v1 | 0.770000 | 0.416000 | 2026-04-06T18:42:39.151981+00:00 |
| distill-format-clean | clean-v1 | distill-format-final | distill-final-next-distill-phase-clean-v1-winner-full_rationale-recipe_mix_c-clean-v1 | 0.770000 | 0.412000 | 2026-04-06T21:28:02.667823+00:00 |
| distill-format-clean | clean-v1 | distill-format-final | distill-final-next-distill-phase-clean-v1-winner-full_rationale-recipe_pure_target-clean-v1 | 0.770000 | 0.408000 | 2026-04-06T15:43:02.833382+00:00 |
| distill-format-clean | clean-v1 | distill-format-heldout | distill-heldout-next-distill-phase-clean-v1-format-answer_only-clean-v1 | 0.844000 | 0.642000 | 2026-04-06T05:53:40.138806+00:00 |
| distill-format-clean | clean-v1 | distill-format-heldout | distill-heldout-next-distill-phase-clean-v1-format-full_rationale-clean-v1 | 0.844000 | 0.640000 | 2026-04-06T11:47:33.974867+00:00 |
| distill-format-clean | clean-v1 | distill-format-heldout | distill-heldout-next-distill-phase-clean-v1-format-short_rationale-clean-v1 | 0.844000 | 0.630000 | 2026-04-06T08:54:32.803525+00:00 |
| distill-format-clean | clean-v1 | distill-format-heldout | distill-heldout-next-distill-phase-clean-v1-winner-full_rationale-recipe_mix_b-clean-v1 | 0.844000 | 0.620000 | 2026-04-06T17:44:36.446940+00:00 |
| distill-format-clean | clean-v1 | distill-format-heldout | distill-heldout-next-distill-phase-clean-v1-winner-full_rationale-recipe_mix_c-clean-v1 | 0.844000 | 0.620000 | 2026-04-06T20:33:59.560234+00:00 |
| distill-format-clean | clean-v1 | distill-format-heldout | distill-heldout-next-distill-phase-clean-v1-winner-full_rationale-recipe_pure_target-clean-v1 | 0.844000 | 0.638000 | 2026-04-06T14:45:48.997215+00:00 |
| final-distill | v2-full-heldout | distill-format-heldout | distill-heldout-next-distill-phase-v2-winner-full_rationale-full-final | 0.743745 | 0.385140 | 2026-04-01T12:24:55.635013+00:00 |
| optimize-loop | v1 | teacher-sft | teacher-sft-c1_e2.0_lr5e-5_r16_s2048-v1 | 0.702047 | 0.347991 | 2026-03-29T02:06:42.293583+00:00 |
| optimize-loop | v1 | teacher-sft | teacher-sft-c2_top1_e1.5_lr3.75e-05_r16_s2048-v1 | 0.721759 | 0.347991 | 2026-03-29T10:38:40.732908+00:00 |
| optimize-loop | v1 | teacher-sft | teacher-sft-c2_top1_e2.0_lr3.75e-05_r16_s2048-v1 | 0.703563 | 0.347991 | 2026-03-29T13:41:53.532412+00:00 |
| qwen-template-fix-v1 | qwen-template-fix-v1 | teacher-sft | teacher-sft-c1_seed_e1.5_lr7.50e-05_s2048_r16_a16_d0.00-qwen-template-fix-v1 | 0.338135 | 0.347991 | 2026-03-31T12:12:54.435566+00:00 |
| teacher-opt | v1 | before-distill | before-distill-teacher-vs-student-base-v1 | 0.740000 | 0.370000 | 2026-03-31T19:10:50.512796+00:00 |