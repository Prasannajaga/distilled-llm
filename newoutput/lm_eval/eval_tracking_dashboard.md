# Eval Tracking Dashboard

- Total eval runs scanned: `4`
- Experiments: `2`
- Teacher optimization runs: `3`

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

## Experiment Timeline

| Experiment | Run ID | Stage | Eval Name | Teacher EM | Student EM | Created UTC |
|---|---|---|---|---:|---:|---|
| before-distill | v2 | before-distill | before-distill-before-distill-v2 | 0.743745 | 0.347991 | 2026-03-28T20:16:11.735109+00:00 |
| optimize-loop | v1 | teacher-sft | teacher-sft-c1_e2.0_lr5e-5_r16_s2048-v1 | 0.702047 | 0.347991 | 2026-03-29T02:06:42.293583+00:00 |
| optimize-loop | v1 | teacher-sft | teacher-sft-c2_top1_e1.5_lr3.75e-05_r16_s2048-v1 | 0.721759 | 0.347991 | 2026-03-29T10:38:40.732908+00:00 |
| optimize-loop | v1 | teacher-sft | teacher-sft-c2_top1_e2.0_lr3.75e-05_r16_s2048-v1 | 0.703563 | 0.347991 | 2026-03-29T13:41:53.532412+00:00 |