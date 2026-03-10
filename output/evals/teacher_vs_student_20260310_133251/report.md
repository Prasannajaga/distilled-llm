# Teacher vs Student Evaluation Report

- Generated at (UTC): `2026-03-10T13:32:51.953132+00:00`
- Eval source: `/home/prasanna/coding/distilled-llm/data/math_eval_100.jsonl`
- Samples: `100`
- Prompt template: `Question: {question} Answer:`
- Decoding: `max_new_tokens=32`, `temperature=0.0`, `top_k=None`, `repetition_penalty=1.0`
- Numeric tolerance: `0.001`

## Side-by-Side Metrics


| Metric                              |                                   Teacher |                                    Student | Better  |
| ----------------------------------- | ----------------------------------------: | -----------------------------------------: | ------- |
| `exact_match_accuracy`              |                                     0.00% |                                      0.00% | tie     |
| `normalized_exact_match_accuracy`   |                                     0.00% |                                      0.00% | tie     |
| `numeric_exact_match_accuracy`      |                                     1.00% |                                      0.00% | teacher |
| `numeric_within_tolerance_accuracy` |                                     1.00% |                                      0.00% | teacher |
| `avg_abs_numeric_error`             | 39,327,191,011,235,949,520,432,922,624.00 | 115,384,615,384,615,370,670,823,440,384.00 | teacher |
| `avg_token_f1`                      |                                  0.000000 |                                   0.000000 | tie     |
| `avg_char_similarity`               |                                  0.049475 |                                   0.056767 | student |
| `answer_extraction_rate`            |                                   100.00% |                                    100.00% | tie     |
| `reference_answer_nll_per_token`    |                                  2.282121 |                                   2.444310 | teacher |
| `reference_answer_perplexity`       |                                  9.797435 |                                     11.523 | teacher |
| `avg_generated_tokens`              |                                    32.000 |                                     32.000 | tie     |
| `truncation_rate`                   |                                   100.00% |                                    100.00% | tie     |
| `avg_latency_ms`                    |                                   372.202 |                                    248.776 | student |
| `tokens_per_second`                 |                                    85.975 |                                    128.630 | student |

## Head-to-Head


| Metric                                    | Value |
| ----------------------------------------- | ----: |
| Teacher wins (correct when student wrong) |     0 |
| Student wins (correct when teacher wrong) |     0 |
| Both correct                              |     0 |
| Both wrong                                |   100 |
| Teacher win rate                          | 0.00% |
| Student win rate                          | 0.00% |

## Why Each Metric Matters


| Metric                              | Why this is important                                                                 | If this improves                                                                                      |
| ----------------------------------- | ------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------- |
| `exact_match_accuracy`              | Strict correctness score: model must output the exact expected answer.                | Higher value means fewer outright wrong final answers and better task reliability.                    |
| `normalized_exact_match_accuracy`   | Correctness after normalization (spacing/punctuation/case). Reduces formatting noise. | Higher value means semantic correctness is improving even when formatting differs.                    |
| `numeric_exact_match_accuracy`      | For numeric tasks, checks exact numeric equality with ground truth.                   | Higher value means stronger arithmetic precision and less numeric hallucination.                      |
| `numeric_within_tolerance_accuracy` | Counts near-correct numeric outputs as correct within tolerance.                      | Higher value means model is converging toward correct magnitudes, even with rounding noise.           |
| `avg_abs_numeric_error`             | Average absolute distance from true numeric answers.                                  | Lower value means typical numeric mistakes are smaller and less harmful.                              |
| `avg_token_f1`                      | Overlap quality between prediction text and reference text.                           | Higher value means better answer content coverage, even when exact strings differ.                    |
| `avg_char_similarity`               | Character-level similarity to reference answer.                                       | Higher value means outputs are structurally closer to targets.                                        |
| `answer_extraction_rate`            | Measures how often model emits an extractable final answer.                           | Higher value means better answer formatting and easier downstream evaluation.                         |
| `reference_answer_nll_per_token`    | How probable the model thinks true answers are, token by token.                       | Lower value means the model distribution aligns better with correct answers.                          |
| `reference_answer_perplexity`       | Exponentiated NLL; interpretable confidence/calibration metric on correct answers.    | Lower value means model is less surprised by correct outputs, typically improving generation quality. |
| `avg_generated_tokens`              | Average answer length generated.                                                      | Lower (without losing accuracy) means more concise answers and lower inference cost.                  |
| `truncation_rate`                   | How often generation hit the max token cap.                                           | Lower value means fewer incomplete answers and better decoding control.                               |
| `avg_latency_ms`                    | Per-sample response time.                                                             | Lower value means faster user experience and higher serving efficiency.                               |
| `tokens_per_second`                 | Generation throughput.                                                                | Higher value means better deployment throughput and lower runtime cost per token.                     |

## Samples Where Models Disagree (up to 20)


| ID | Question                            | Gold | Teacher | Student |
| -: | ----------------------------------- | ---- | ------- | ------- |
|  - | No disagreement samples in this run | -    | -       | -       |
