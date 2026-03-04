#!/usr/bin/env bash
set -euo pipefail

if command -v python >/dev/null 2>&1; then
  PYTHON_BIN="python"
elif command -v python3 >/dev/null 2>&1; then
  PYTHON_BIN="python3"
else
  echo "No python interpreter found (expected python or python3)." >&2
  exit 1
fi

DATA_BIN_DIR="${DATA_BIN_DIR:-/tmp/vertex-bins}"
DATA_BIN_NAME="${DATA_BIN_NAME:-data.bin}"
DATA_BIN_PATH="${DATA_BIN_PATH:-${DATA_BIN_DIR}/${DATA_BIN_NAME}}"
BLOCK_SIZE="${BLOCK_SIZE:-512}"
DATA_PACK_BATCH_SIZE="${DATA_PACK_BATCH_SIZE:-1024}"
DATA_NUM_WORKERS="${DATA_NUM_WORKERS:-$(nproc)}"
TOKENIZER_MODEL="${TOKENIZER_MODEL:-TinyLlama/TinyLlama-1.1B-Chat-v1.0}"
DATASETS_JSON="${DATASETS_JSON:-}"
NPROC_PER_NODE="${NPROC_PER_NODE:-${WORLD_SIZE:-1}}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

echo "========================================="
echo " Step 1: Build Single Packed Bin"
echo "========================================="

LOAD_ARGS=(
  --output_dir "$DATA_BIN_DIR"
  --output_bin_name "$DATA_BIN_NAME"
  --block_size "$BLOCK_SIZE"
  --pack_batch_size "$DATA_PACK_BATCH_SIZE"
  --num_workers "$DATA_NUM_WORKERS"
  --tokenizer_model "$TOKENIZER_MODEL"
)

if [ -n "$DATASETS_JSON" ]; then
  LOAD_ARGS+=(--datasets_json "$DATASETS_JSON")
fi

"$PYTHON_BIN" -m scripts.loadDataset "${LOAD_ARGS[@]}"

echo ""
echo "========================================="
echo " Step 2: Distributed Training"
echo "========================================="
echo "Using bin: $DATA_BIN_PATH"
echo "nproc_per_node: $NPROC_PER_NODE"

"$PYTHON_BIN" -m torch.distributed.run \
  --nproc_per_node="$NPROC_PER_NODE" \
  --module scripts.train_vertex \
  --bin_path "$DATA_BIN_PATH" \
  "$@"

echo ""
echo "========================================="
echo " Pipeline Complete"
echo "========================================="
