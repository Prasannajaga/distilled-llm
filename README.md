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

At a high level, the repo is split into 4 layers.

### 1) Model layer

- `scripts/model.py`
- `utils/transformer.py`

Contains:

- GQA + RoPE attention blocks
- RMSNorm + SwiGLU FFN
- KV-cache aware generation
- recent optimizations (QK normalization, FFN alignment)

### 2) Data layer

- `utils/packed_dataset_builder.py`
- `scripts/loadDataset.py` (called in pipeline)
- `Cdatasets/*`

Flow:

1. load HF dataset(s)
2. tokenize
3. pack into fixed block-size chunks
4. save packed bin/index
5. memory-map for training dataloaders

### 3) Training layer

- `utils/trainer.py`
- `scripts/train_pretrain.py`
- `scripts/train_distill.py`
- `scripts/train_vertex.py`

Covers:

- mixed precision
- gradient accumulation
- checkpoint/resume
- optional distributed training (DDP)
- Vertex experiment hooks/logging

### 4) Deployment/ops layer

- `deploy.py`
- `scripts/step-train.sh`
- `scripts/run_step_train.py`

Covers:

- package + upload source dist
- submit Vertex custom job
- run worker-side data packing then training

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

## Key Files to Start With

- `scripts/model.py`: model architecture and generation
- `utils/transformer.py`: attention/FFN/norm primitives
- `utils/trainer.py`: train loop + checkpointing
- `deploy.py`: Vertex submission entrypoint
- `scripts/step-train.sh`: worker-side step pipeline

## Notes

- Keep `block_size` consistent across packing and training.
- For large dataset packing on Vertex, disk sizing matters a lot.
- For deeper progress details, read [progress.md](./progress.md).
- For failures/fixes during deployment, read [deploy.md](./deploy.md).
