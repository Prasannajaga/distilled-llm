# Metrics Summary: mini-math-teacher-v1-30K

Folder: `/home/prasanna/coding/distilled-llm/output/mini-math-teacher-v1-30K/metrics`
Source files:
- `final_ckpt_30000.json`
- `results.json`
- `metrics_prompt_metrics_step_30000.json`

## 1) Quick Snapshot

| Item | Value |
|---|---:|
| Model | GQATransformer |
| Parameters | 146,790,432 (100% trainable) |
| Steps completed | 30,000 / 30,000 |
| Train time | 139,166.93s (~38.66h) |
| Tokens trained (global est.) | 4,419,360,000 |
| Throughput (global est.) | 31,755.821 tok/s |
| Best train loss | 1.7606 (step 11,720) |
| Final train loss | 3.5598 |
| Best/final val loss | 3.2709 (step 30,000) |
| Hardware | 2 GPUs, peak 10.635 GB |

## 2) Model Details

| Field | Value |
|---|---:|
| Architecture | `GQATransformer` |
| Attention used | Grouped Query Attention (inferred from model name) |
| Layers (`n_layer`) | 18 |
| Embedding dim (`n_embd`) | 768 |
| Heads (`n_head`) | 16 |
| Context length (`block_size`) | 1024 |
| Optimizer | AdamW |
| Learning rate | 1.2e-4 |
| Weight decay | 0.1 |
| Grad accumulation | 12 |
| Effective batch size (global) | 144 |
| Mixed precision | bfloat16 AMP |
| Gradient checkpointing | true |

## 3) Dataset Details

### Source mix (by row count)

| Dataset | Rows | Share |
|---|---:|---:|
| `HuggingFaceFW/fineweb` (`sample-10BT`) | 8,363,014 | 56.75% |
| `open-web-math/open-web-math` | 6,315,233 | 42.85% |
| `incredible45/Gutenberg-BookCorpus-Cleaned-Data-English` | 58,653 | 0.40% |

Simple visualization:
- FineWeb: `############################` 56.75%
- OpenWebMath: `#####################` 42.85%
- Gutenberg/BookCorpus: `.` 0.40%

### Packed/training stats

| Metric | Value |
|---|---:|
| Packed total tokens | 27,315,400,704 |
| Total sequences | 26,675,196 |
| Avg tokens per sequence | 1,024 |
| Train split samples | 21,340,157 (80.00%) |
| Val split samples | 5,335,039 (20.00%) |
| Tokens seen in this run (global est.) | 4,419,360,000 |
| Coverage of packed tokens | 16.18% |

## 4) Evals (Prompt Metrics @ step 30k)

### Eval setup

| Field | Value |
|---|---:|
| Eval prompts | 25 |
| Successful calls | 25 |
| Failed calls | 0 |
| `max_new_tokens` | 512 |
| Eval file timestamp | 2026-03-06T22:46:36.482649Z |

### Performance summary

| Metric | Min | P50 | Avg | Max |
|---|---:|---:|---:|---:|
| Generated tokens | 14 | 256 | 310.68 | 512 |
| Latency (ms) | 504.01 | 7,617.65 | 9,284.55 | 15,338.27 |
| Tokens/sec | 27.78 | 33.51 | 32.90 | 33.90 |

### Truncation / long-generation signal

| Metric | Value |
|---|---:|
| Prompts hitting generation cap (`512`) | 11 / 25 |
| Cap-hit rate | 44.00% |

Visualization:
- Hit cap: `###########..............` (44%)
- Not hit: `##############...........` (56%)

### Category breakdown

| Group | Prompts | Avg generated tokens | Avg latency (ms) | Avg tok/s | Cap hits |
|---|---:|---:|---:|---:|---:|
| Math (idx 0-9) | 10 | 253.8 | 7,634.39 | 32.74 | 3 |
| Tech/General (idx 10-19) | 10 | 386.3 | 11,521.14 | 32.75 | 6 |
| Creative/Writing (idx 20-24) | 5 | 273.2 | 8,111.70 | 33.51 | 2 |

## 5) Quality Notes

- Inference reliability is high (25/25 requests succeeded).
- Output quality appears inconsistent: multiple completions are off-task or drift from the prompt intent.
- High cap-hit rate suggests frequent non-natural stopping (responses often run to limit).

## 6) Consistency Checks

- Core run/loss values match between `final_ckpt_30000.json` and `results.json`.
- Artifact paths in `results.json` point to `/outputs/mini-code-v1/...` while this run directory is `mini-math-teacher-v1-30K`; likely naming/path carryover.

## 7) Bottom Line

Training infrastructure metrics look healthy and the run completed fully. Prompt-based evaluation indicates a quality gap (instruction following/content relevance) despite successful generation calls.
