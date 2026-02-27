#!/usr/bin/env bash
set -euo pipefail

OPENWEBMATH_DIR="./data/AutoMathText"
NUMINAMATH_DIR="./data/numinamath"
PRETRAIN_CKPT_DIR="./checkpoints/pretrain"
DISTILL_CKPT_DIR="./checkpoints/distill"

echo "========================================="
echo " Step 1: Dataset Preparation"
echo "========================================="
 
echo "[DOWNLOAD] Fetching OpenWebMath..."
python -c "
from Cdatasets.dataset import download_dataset
download_dataset('math-ai/AutoMathText', '$OPENWEBMATH_DIR', split='train', subset='arxiv-0.60-to-1.00', skip=True)
"
fi

# if [ -d "$NUMINAMATH_DIR" ] && [ "$(ls -A "$NUMINAMATH_DIR" 2>/dev/null)" ]; then
#     echo "[SKIP] NuminaMath-CoT already exists at $NUMINAMATH_DIR"
# else
#     echo "[DOWNLOAD] Fetching NuminaMath-CoT..."
#     python -c "
# from scripts.dataset import download_dataset
# download_dataset('AI-MO/NuminaMath-CoT', '$NUMINAMATH_DIR', split='train[:5000]')
# "
# fi

# echo ""
# echo "========================================="
# echo " Step 1.5: Tokenizer Training"
# echo "========================================="

# TOKENIZER_DIR="./tokenizer_artifacts"
# if [ -f "$TOKENIZER_DIR/tokenizer.json" ]; then
#     echo "[SKIP] Tokenizer already exists at $TOKENIZER_DIR/tokenizer.json"
# else
#     echo "[TRAIN] Training BPE tokenizer on OpenWebMath (shards 0-2)..."
#     python -m scripts.train_tokenizer \
#         --vocab_size 8192 \
#         --dataset_name "open-web-math/open-web-math" \
#         --dataset_split "train" \
#         --data_files \
#             "data/train-00000-of-00114-*.parquet" \
#             "data/train-00001-of-00114-*.parquet" \
#             "data/train-00002-of-00114-*.parquet" \
#         --output_dir "$TOKENIZER_DIR"
# fi

echo ""
echo "========================================="
echo " Step 2: Pre-training (Phase 1)"
echo "========================================="

# python -m scripts.train_pretrain \
#     --batch_size 16 \
#     --lr 3e-4 \
#     --weight_decay 0.1 \
#     --epochs 1 \
#     --warmup_steps 500 \
#     --grad_clip 1.0 \
#     --log_interval 50 \
#     --save_interval 1000 \
#     --block_size 512 \
#     --dataset_path "$OPENWEBMATH_DIR" \
#     --checkpoint_dir "$PRETRAIN_CKPT_DIR"

# echo ""
# echo "========================================="
# echo " Step 3: Knowledge Distillation (Phase 2)"
# echo "========================================="

# python -m scripts.train_distill \
#     --batch_size 16 \
#     --lr 5e-5 \
#     --weight_decay 0.1 \
#     --epochs 1 \
#     --warmup_steps 200 \
#     --grad_clip 1.0 \
#     --log_interval 50 \
#     --save_interval 1000 \
#     --block_size 512 \
#     --dataset_path "$NUMINAMATH_DIR" \
#     --teacher_checkpoint "$PRETRAIN_CKPT_DIR/latest_checkpoint.pt" \
#     --student_checkpoint "$PRETRAIN_CKPT_DIR/latest_checkpoint.pt" \
#     --temp 3.0 \
#     --alpha_ce 0.2 \
#     --alpha_kl 0.5 \
#     --alpha_mse 0.3 \
#     --checkpoint_dir "$DISTILL_CKPT_DIR"

echo ""
echo "========================================="
echo " Pipeline Complete"
echo "========================================="
echo "Pre-train checkpoints: $PRETRAIN_CKPT_DIR"
echo "Distill checkpoints:   $DISTILL_CKPT_DIR"
