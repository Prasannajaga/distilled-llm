# Deployment Notes (Last 2 Days)

This document captures the deployment flow we followed, what failed, how we fixed it, and the CLI commands used.

## Quick Summary

In the last 2 days, we moved from basic script-level training to a repeatable Vertex AI pipeline:

1. package project
2. submit Vertex custom job
3. run dataset packing in worker
4. run DDP training
5. save checkpoints locally and sync artifacts to GCS

Key progress commits in this window:

- `batch scripts installation`
- `model enhancement`
- `deploy vertex setup`
- `presets update`

## Deployment Flow We Used

### 1) Build + upload package + submit Vertex job

```bash
python deploy.py \
  --project_id <gcp-project> \
  --region us-central1 \
  --bucket_uri gs://<bucket> \
  --machine_type g2-standard-24 \
  --accelerator_type NVIDIA_L4 \
  --accelerator_count 2 \
  --replica_count 1 \
  --boot_disk_size 300 \
  --nproc_per_node 2 \
  --display_name distilled-llm-train
```

`deploy.py` does this sequence:

1. `python setup.py sdist --formats=gztar`
2. upload tarball to `gs://<bucket>/packages/...`
3. submit Vertex `CustomJob` with env vars for data packing + training

### 2) Worker entrypoint pipeline

Inside Vertex worker, `scripts/step-train.sh` runs:

```bash
python -m scripts.loadDataset ...
python -m torch.distributed.run --nproc_per_node=<N> --module scripts.train_vertex --bin_path <packed_bin>
```

### 3) Data packing defaults used

Important env/defaults in `scripts/step-train.sh`:

- `DATA_BIN_DIR=/tmp/vertex-bins`
- `DATA_BIN_NAME=data.bin`
- `BLOCK_SIZE=512`
- `DATA_PACK_BATCH_SIZE=1024`
- `DATA_NUM_WORKERS=$(nproc)`
- `TOKENIZER_MODEL=TinyLlama/TinyLlama-1.1B-Chat-v1.0`
- `NPROC_PER_NODE=${WORLD_SIZE:-1}`

## Failures We Faced and How We Fixed Them

### Failure 1: Disk pressure during packing

Symptom:

- dataset mixture and intermediate packing consumed very high disk quickly.

Fix:

- increased boot disk (`--boot_disk_size 300`)
- used packed `.bin` flow instead of repeated raw-tokenization every time
- added bucket-aware dataset cache flags (`ENABLE_BUCKET`, `DATASET_NAME`, `BUCKET_DATASET_PATH`) to avoid unnecessary rebuilds.

### Failure 2: DDP process/GPU mismatch

Symptom:

- unstable distributed startup or poor utilization when process count did not match GPU count.

Fix:

- set `--nproc_per_node` to accelerator count
- propagated this through deploy env and `torch.distributed.run` call.

### Failure 3: Noisy logs looked like hard errors

Symptom:

- pip and warning noise in Vertex logs made triage harder.

Fix:

- set envs in job spec to suppress noisy pip warnings:
  - `PIP_NO_WARN_SCRIPT_LOCATION=1`
  - `PIP_ROOT_USER_ACTION=ignore`
  - `PIP_DISABLE_PIP_VERSION_CHECK=1`

### Failure 4: Slow restart loop after interruption

Symptom:

- interrupted runs had expensive restart time.

Fix:

- checkpoint/resume flow used in trainer
- optimizer/scaler persisted for resume checkpoints
- final checkpoint can be lighter when full optimizer state is not required.

### Failure 5: Data stage dominated training wall-clock

Symptom:

- tokenization + packing took longer than expected compared to pure training.

Fix:

- standardized pipeline: dataset -> tokenize -> pack -> mmap -> dataloader
- made the data build step explicit in deployment pipeline.

## Most Useful CLI Snippets

### Submit with args file

```bash
python deploy.py @deploy.args
```

### Force non-default dataset JSON

```bash
python deploy.py \
  --project_id <gcp-project> \
  --region us-central1 \
  --bucket_uri gs://<bucket> \
  --datasets_json ./datasets.json
```

### Local dry run of worker pipeline

```bash
bash scripts/step-train.sh --epochs 1 --total_steps 50
```

### Manual train module run

```bash
python -m scripts.train_vertex --bin_path /tmp/vertex-bins/data.bin
```

## What Moved Us Forward

- standardized packaging + submission in one command (`deploy.py`)
- reliable worker pipeline (`scripts/step-train.sh`)
- explicit resource sizing for large corpora
- resumable training and cleaner logs
- better observability separation (training logic vs experiment logging)
