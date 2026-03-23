# GSM8K Teacher vs Student (Before Distillation)

- Generated at (UTC): `2026-03-23T14:42:47.836203+00:00`
- Dataset source: `/media/prasanna/716F26140AED9B67/datasets/openai___gsm8k/main/0.0.0/cc7b047b6e5bb11b4f1af84efc572db110a51b3c/gsm8k-test.arrow`
- Split: `test`
- Evaluated samples: `10`
- Requested max samples: `10`
- Teacher model: `Qwen/Qwen2-Math-1.5B-Instruct`
- Student model: `Qwen/Qwen2-0.5B-Instruct`

## Decoding

| Key | Value |
|---|---|
| `max_new_tokens` | `128` |
| `temperature` | `0.0` |
| `top_p` | `1.0` |
| `repetition_penalty` | `1.0` |
| `render_mode` | `auto` |
| `numeric_tolerance` | `0.001` |

## Side-by-Side Metrics

| Metric | Teacher | Student |
|---|---:|---:|
| `normalized_exact_match_accuracy` | 0.00% | 0.00% |
| `numeric_within_tolerance_accuracy` | 0.00% | 40.00% |
| `avg_abs_numeric_error` | 11,736.00 | 6,257.58 |
| `avg_generated_tokens` | 128.000 | 108.600 |
| `truncation_rate` | 100.00% | 60.00% |
| `avg_latency_ms` | 2,721.63 | 1,085.01 |
| `tokens_per_second` | 47.031 | 100.092 |

## Head-to-Head

| Metric | Value |
|---|---:|
| Teacher wins | 0 |
| Student wins | 0 |
| Both correct | 0 |
| Both wrong | 10 |
| Teacher win rate | 0.00% |
| Student win rate | 0.00% |
