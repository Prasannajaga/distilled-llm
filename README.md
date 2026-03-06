# Distilled LLM (Teacher -> Student)

A practical project to learn and build end-to-end LLM distillation:

- dataset ingestion + packing
- pretraining and distillation loops
- local and Vertex AI training
- inference and deployment utilities

For detailed journey notes and deployment history:

- [progress.md](./progress.md)
- [deploy.md](./deploy.md)

## Project Architecture

```text
.
├── deploy.py                 # Vertex submission entrypoint
├── presets/                  # Preconfigured deployment and training templates
├── scripts/                  # Executable training and inference scripts
│   ├── infer.py              # Local inference testing
│   ├── loadDataset.py        # Dataset loading & manipulation
│   ├── model.py              # Model definitions and KV-cache aware generation
│   ├── run_step_train.py     # Worker-side execution wrapper
│   ├── step-train.sh         # Worker pipeline entry
│   ├── train_distill.py      # Distillation training loop
│   ├── train_pretrain.py     # Pretraining loop
│   └── train_vertex.py       # Vertex AI training wrapper
└── utils/                    # Core utilities and modules
    ├── config.py             # Shared project configurations
    ├── engine.py             # Inference/generation engine
    ├── packed_dataset_builder.py  # Data packing logic
    ├── trainer.py            # Main training loop and checkpointing
    └── transformer.py        # Transformer primitives (GQA, RMSNorm, etc.)
```

## Simple Usage

### 1) Local pretraining

```bash
python -m scripts.train_pretrain \
  --bin_path ./data/pretrain_tokens.bin \
  --epochs 1
```

### 2) Local inference

```bash
python -m scripts.infer --output ./output/<model-dir> "hello"
```

### 3) Vertex deployment

```bash
python deploy.py \
  --project_id <gcp-project> \
  --region us-central1 \
  --bucket_uri gs://<bucket> \
  --machine_type g2-standard-24 \
  --accelerator_type NVIDIA_L4 \
  --accelerator_count 2 \
  --boot_disk_size 300
```

### 4) Worker pipeline entry

```bash
bash scripts/step-train.sh --epochs 1 --total_steps 100
```
