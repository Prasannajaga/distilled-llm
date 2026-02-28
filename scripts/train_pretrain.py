from __future__ import annotations

import argparse
from functools import partial
from pathlib import Path
from typing import Final

from torch.utils.data import DataLoader

from utils.common import get_device
from utils.config import TrainingConfig
from Cdatasets.dataset import pretrain_collate_fn
from utils.binary_dataset import TokenizedBinaryDataset
from scripts.model import GQATransformer
from Cdatasets.tokenizer import load_tokenizer
from utils.trainer import Trainer

TINYLLAMA_MODEL_NAME: Final[str] = "TinyLlama/TinyLlama-1.1B-Chat-v1.0" 

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase 1: Pre-training")

    p.add_argument("--dataset_path", type=str, required=True)
    p.add_argument("--bin_path", type=str, default="./data/pretrain_tokens.bin")
    p.add_argument("--max_rows", type=int, default=None)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--teacher", action="store_true", default=False)

    p.add_argument("--device", type=str)
    p.add_argument("--seed", type=int)
    p.add_argument("--num_workers", type=int)

    p.add_argument("--n_layer", type=int)
    p.add_argument("--n_embd", type=int)
    p.add_argument("--n_head", type=int)
    p.add_argument("--block_size", type=int)

    p.add_argument("--train_batch_size", type=int)
    p.add_argument("--eval_batch_size", type=int)
    p.add_argument("--grad_accum_steps", type=int)
    p.add_argument("--validation_batch_count", type=int)

    p.add_argument("--optimizer", type=str)
    p.add_argument("--lr", type=float)
    p.add_argument("--weight_decay", type=float)
    p.add_argument("--grad_clip_norm", type=float)
    p.add_argument("--eps", type=float)

    p.add_argument("--enable_gradient_checkpointing", type=bool)
    p.add_argument("--amp_dtype", type=str)
    p.add_argument("--clear_cache_interval", type=int)

    p.add_argument("--total_steps", type=int)
    p.add_argument("--warmup_steps", type=int)

    p.add_argument("--ckpt_dir", type=str)
    p.add_argument("--ckpt_interval_steps", type=int)

    p.add_argument("--log_interval_steps", type=int)

    return p.parse_args()


def apply_overrides(config: TrainingConfig, args: argparse.Namespace) -> None:
    for key, value in vars(args).items():
        if value is not None and hasattr(config, key):
            setattr(config, key, value)
 


def main() -> None:
    args = parse_args()
    config = TrainingConfig()
    apply_overrides(config, args)

    device = get_device(config.device)
    print(f"[DEVICE] Using {device}")

    tokenizer = load_tokenizer(TINYLLAMA_MODEL_NAME)

    if args.teacher: 
        role = "Teacher"
        prefix = "teacher_pretrain"
    else:
        role = "Student"
        prefix = "student_pretrain"

    model = GQATransformer(
        num_layers=config.n_layer,
        n_emb=config.n_embd,
        n_head=config.n_head,
        n_kv_head=config.n_head // 2,
        vocab_size=len(tokenizer),
        block_size=config.block_size,
    )

    trainer = Trainer(
        model=model,
        config=config,
        device=device,
        tokenizer=tokenizer,
        ckpt_dir=config.ckpt_dir,
    )

    checkpoint_dir = Path(config.ckpt_dir)
    if checkpoint_dir.exists():
        from utils.common import load_checkpoint
        step, _ = load_checkpoint(
            checkpoint_dir, model, trainer.optimizer, device, prefix=prefix
        )
        if step > 0:
            trainer.global_step = step

    dataset = TokenizedBinaryDataset(
        source=args.dataset_path,
        tokenizer=tokenizer,
        block_size=config.block_size,
        bin_path=args.bin_path,
        max_rows=args.max_rows,
    )
    
    collate = partial(pretrain_collate_fn, pad_id=tokenizer.pad_token_id)
    dataloader = DataLoader(
        dataset,
        batch_size=config.train_batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        collate_fn=collate,
        drop_last=True,
    )

    trainer.log_dataset_info(dataset, dataloader, config)
    
    print(f"[TRAIN] Starting {role} pre-training from step {trainer.global_step}")

    trainer.train(
        train_dataloader=dataloader,
        epochs=args.epochs,
        max_steps=config.total_steps
    )

    print(f"[DONE] {role} pre-training complete.")


if __name__ == "__main__":
    main()
