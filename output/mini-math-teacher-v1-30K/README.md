# mini-math-teacher-v1-base-30K

`mini-math-teacher-v1-base-30K` is a 146.8M-parameter causal language model checkpoint trained for 30k steps with a GQA-based transformer architecture.

`base model:` model.safetensors
`checkpoint model:` checkpoints_model_30000.safetensors (resume training from this optimizer & scaler)

## Model Details

Field | Value |
---|---:|
Model type | Causal LM (`GQATransformer`) |
Attention | Grouped Query Attention |
Parameters | 146,790,432 (100% trainable) |
Layers | 18 |
Hidden size | 768 |
Attention heads | 16 |
Context length | 1024 |
Precision during training | bfloat16 AMP |
Optimizer | AdamW |
Learning rate | 1.2e-4 |
Weight decay | 0.1 |
Epsilon | 1e-8 |
Gradient clipping | 1.0 |
Gradient accumulation | 12 |
Effective global batch size | 144 |
Gradient checkpointing | enabled |

## Training Details

Field | Value |
|---|---:|
Total steps | 30,000 |
World size | 2 GPUs |
Training time | 139,166.93 seconds (~38.66 hours) |
Tokens trained (local) | 2,209,680,000 |
Tokens trained (global estimate) | 4,419,360,000 | 
Best train loss | 2.7606 |
Final train loss | 3.5598 |
Worst train loss | 10.5604 |
Best val loss | 3.2709 (step 30,000) |
Final val loss | 3.2709 |

## Dataset Details

Training data is a blend of three sources:

| Dataset | Split | Text column | Rows | Share |
|---|---|---|---:|---:|
| `HuggingFaceFW/fineweb` (`sample-10BT`) | train | `text` | 8,363,014 | 56.75% |
| `open-web-math/open-web-math` | train | `text` | 6,315,233 | 42.85% |
| `incredible45/Gutenberg-BookCorpus-Cleaned-Data-English` | train | `context` | 58,653 | 0.40% |

Packed data and split stats:
- Packed total tokens: 27,315,400,704
- Packed total sequences: 26,675,196
- Avg packed sequence length: 1,024
- Train split samples: 21,340,157
- Val split samples: 5,335,039

## Intended Use

- Research and experimentation with small-to-mid scale GQA language modeling.
- Baseline for instruction tuning, preference tuning, or decoding strategy experiments.
- Not recommended for production use without additional alignment and benchmark validation.

## Limitations

- Prompt benchmark is small (25 prompts) and not a standardized leaderboard suite.
- `success=true` indicates inference success, not answer correctness.
- Quality on instruction-following appears inconsistent from sampled completions.

## Risks and Safety

This model may generate incorrect, irrelevant, or misleading content. Validate outputs before use in any high-stakes setting (education, legal, medical, financial, or safety-critical workflows).
 
