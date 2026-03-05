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
ENABLE_BUCKET="${ENABLE_BUCKET:-1}"
BUCKET_URI="${BUCKET_URI:-}"
DATASET_NAME="${DATASET_NAME:-default}"
BUCKET_DATASET_PATH="${BUCKET_DATASET_PATH:-}"
NPROC_PER_NODE="${NPROC_PER_NODE:-${WORLD_SIZE:-1}}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

print_system_specs() {
  echo "========================================="
  echo " System Specs"
  echo "========================================="

  # CPU
  local cpu_model cpu_cores
  cpu_model="$(grep -m1 'model name' /proc/cpuinfo 2>/dev/null | cut -d: -f2- | xargs || true)"
  cpu_cores="$(nproc 2>/dev/null || true)"
  if [ -n "${cpu_model:-}" ] || [ -n "${cpu_cores:-}" ]; then
    echo "CPU: ${cpu_model:-unknown} | Cores: ${cpu_cores:-unknown}"
  else
    echo "CPU: unknown"
  fi

  # RAM
  local mem_total_kb mem_avail_kb
  mem_total_kb="$(awk '/MemTotal/ {print $2}' /proc/meminfo 2>/dev/null || true)"
  mem_avail_kb="$(awk '/MemAvailable/ {print $2}' /proc/meminfo 2>/dev/null || true)"
  if [ -n "${mem_total_kb:-}" ] && [ -n "${mem_avail_kb:-}" ]; then
    local mem_total_gb mem_avail_gb
    mem_total_gb="$(awk "BEGIN {printf \"%.2f\", ${mem_total_kb}/1024/1024}")"
    mem_avail_gb="$(awk "BEGIN {printf \"%.2f\", ${mem_avail_kb}/1024/1024}")"
    echo "RAM: ${mem_total_gb} GiB total | ${mem_avail_gb} GiB available"
  else
    echo "RAM: unknown"
  fi

  # SSD / Disk
  local root_src root_base root_rotational root_class
  root_src="$(df -P / 2>/dev/null | awk 'NR==2 {print $1}' || true)"
  root_base="$(basename "${root_src:-}" | sed -E 's/p?[0-9]+$//' || true)"
  root_rotational=""
  if command -v lsblk >/dev/null 2>&1 && [ -n "${root_base:-}" ] && [ -e "/dev/${root_base}" ]; then
    root_rotational="$(lsblk -ndo ROTA "/dev/${root_base}" 2>/dev/null | head -n1 | tr -d '[:space:]' || true)"
  fi
  case "${root_rotational:-}" in
    0) root_class="SSD" ;;
    1) root_class="HDD" ;;
    *) root_class="unknown" ;;
  esac
  echo "Storage Type (/): ${root_class}"
  df -h / 2>/dev/null | awk 'NR==2 {printf "Storage (/): %s total | %s used | %s avail\\n", $2, $3, $4}' || true

  if [ -n "${DATA_BIN_DIR:-}" ]; then
    mkdir -p "$DATA_BIN_DIR" || true
    df -h "$DATA_BIN_DIR" 2>/dev/null | awk 'NR==2 {printf "Storage (DATA_BIN_DIR): %s total | %s used | %s avail\\n", $2, $3, $4}' || true
  fi
  echo ""
}

print_system_specs

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
  --enable_bucket "$ENABLE_BUCKET"
  --dataset_name "$DATASET_NAME"
)

if [ -n "$DATASETS_JSON" ]; then
  LOAD_ARGS+=(--datasets_json "$DATASETS_JSON")
fi
if [ -n "$BUCKET_URI" ]; then
  LOAD_ARGS+=(--bucket_uri "$BUCKET_URI")
fi
if [ -n "$BUCKET_DATASET_PATH" ]; then
  LOAD_ARGS+=(--bucket_dataset_path "$BUCKET_DATASET_PATH")
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
