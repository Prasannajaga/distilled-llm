from __future__ import annotations

import argparse
import math
import os
import sys
from functools import partial
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from utils.config import (
    student_config,
    teacher_config,
    default_distill_config,
    distill_train_config,
)
from Cdatasets.dataset import FineTuneDataset, finetune_collate_fn
from scripts.model import GQATransformer
from Cdatasets.tokenizer import MathTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 2: Knowledge Distillation")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--warmup_steps", type=int, default=200)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--log_interval", type=int, default=50)
    parser.add_argument("--save_interval", type=int, default=1000)
    parser.add_argument("--block_size", type=int, default=512)
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints/distill")
    parser.add_argument("--student_checkpoint", type=str, default=None)
    parser.add_argument("--teacher_checkpoint", type=str, required=True)
    parser.add_argument("--temp", type=float, default=3.0)
    parser.add_argument("--alpha_ce", type=float, default=0.2)
    parser.add_argument("--alpha_kl", type=float, default=0.5)
    parser.add_argument("--alpha_mse", type=float, default=0.3)
    parser.add_argument("--device", type=str, default="auto")
    return parser.parse_args()


def get_device(requested: str) -> torch.device:
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(requested)


def get_lr(step: int, warmup_steps: int, max_lr: float, total_steps: int) -> float:
    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps
    decay_ratio = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return max_lr * max(coeff, 0.1)


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    epoch: int,
    loss: float,
    checkpoint_dir: Path,
) -> None:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    state = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "step": step,
        "epoch": epoch,
        "loss": loss,
    }
    path = checkpoint_dir / "latest_checkpoint.pt"
    torch.save(state, path)
    step_path = checkpoint_dir / f"checkpoint_step_{step}.pt"
    torch.save(state, step_path)
    print(f"[CHECKPOINT] Saved at step {step} to {path}")


def load_checkpoint(
    checkpoint_dir: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[int, int]:
    path = checkpoint_dir / "latest_checkpoint.pt"
    if not path.exists():
        print("[INIT] No distillation checkpoint found. Starting from scratch.")
        return 0, 0

    print(f"[RESUME] Loading checkpoint from {path}")
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint["step"], checkpoint["epoch"]


def load_student_from_pretrain(
    model: nn.Module,
    pretrain_checkpoint: str,
    device: torch.device,
) -> None:
    path = Path(pretrain_checkpoint)
    if not path.exists():
        raise FileNotFoundError(f"Student pretrain checkpoint not found: {path}")
    print(f"[LOAD] Loading pre-trained student weights from {path}")
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])


def compute_distillation_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    targets: torch.Tensor,
    student_hidden: torch.Tensor,
    teacher_hidden: torch.Tensor,
    regressor: nn.Linear,
    temperature: float,
    alpha_ce: float,
    alpha_kl: float,
    alpha_mse: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    B, T, V_student = student_logits.shape
    _, _, V_teacher = teacher_logits.shape
    V_min = min(V_student, V_teacher)

    ce_loss = F.cross_entropy(
        student_logits.view(B * T, V_student),
        targets.view(B * T),
        ignore_index=-100,
    )

    student_soft = F.log_softmax(student_logits[:, :, :V_min] / temperature, dim=-1)
    teacher_soft = F.softmax(teacher_logits[:, :, :V_min] / temperature, dim=-1)
    kl_loss = F.kl_div(
        student_soft.view(-1, V_min),
        teacher_soft.view(-1, V_min),
        reduction="batchmean",
    ) * (temperature ** 2)

    projected_student = regressor(student_hidden)
    mse_loss = F.mse_loss(projected_student, teacher_hidden.detach())

    total_loss = alpha_ce * ce_loss + alpha_kl * kl_loss + alpha_mse * mse_loss

    metrics = {
        "ce_loss": ce_loss.item(),
        "kl_loss": kl_loss.item(),
        "mse_loss": mse_loss.item(),
        "total_loss": total_loss.item(),
    }
    return total_loss, metrics


def train(args: argparse.Namespace) -> None:
    device = get_device(args.device)
    print(f"[DEVICE] Using {device}")

    tokenizer = MathTokenizer()
    vocab = tokenizer.vocab_size

    t_cfg = teacher_config(vocab_size=vocab)
    s_cfg = student_config(vocab_size=vocab)

    teacher = GQATransformer(
        num_layers=t_cfg.num_layers,
        n_emb=t_cfg.n_embd,
        n_head=t_cfg.n_head,
        n_kv_head=t_cfg.n_kv_head,
        vocab_size=t_cfg.vocab_size,
        block_size=args.block_size,
        dropout=0.0,
    ).to(device)

    print(f"[LOAD] Loading teacher checkpoint from {args.teacher_checkpoint}")
    teacher_ckpt = torch.load(
        args.teacher_checkpoint, map_location=device, weights_only=False
    )
    teacher.load_state_dict(teacher_ckpt["model_state_dict"])
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False
    print(f"[TEACHER] {sum(p.numel() for p in teacher.parameters()) / 1e6:.2f}M params (frozen)")

    student = GQATransformer(
        num_layers=s_cfg.num_layers,
        n_emb=s_cfg.n_embd,
        n_head=s_cfg.n_head,
        n_kv_head=s_cfg.n_kv_head,
        vocab_size=s_cfg.vocab_size,
        block_size=args.block_size,
        dropout=s_cfg.dropout,
    ).to(device)

    if args.student_checkpoint:
        load_student_from_pretrain(student, args.student_checkpoint, device)

    regressor = nn.Linear(s_cfg.n_embd, t_cfg.n_embd, bias=False).to(device)

    student_params = sum(p.numel() for p in student.parameters())
    regressor_params = sum(p.numel() for p in regressor.parameters())
    print(f"[STUDENT] {student_params / 1e6:.2f}M params + {regressor_params / 1e6:.2f}M regressor")

    all_params = list(student.parameters()) + list(regressor.parameters())
    optimizer = torch.optim.AdamW(
        all_params,
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.95),
    )

    checkpoint_dir = Path(args.checkpoint_dir)
    global_step, start_epoch = load_checkpoint(checkpoint_dir, student, optimizer, device)

    dataset = FineTuneDataset(
        data_dir=args.dataset_path,
        tokenizer=tokenizer,
        block_size=args.block_size,
    )
    collate = partial(finetune_collate_fn, pad_id=tokenizer.pad_id)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collate,
        drop_last=True,
    )

    total_steps = args.epochs * len(dataloader)
    print(f"[TRAIN] Total steps: {total_steps}, starting from step {global_step}")

    student.train()
    regressor.train()

    for epoch in range(start_epoch, args.epochs):
        for batch_idx, (input_ids, targets, attention_mask) in enumerate(dataloader):
            if global_step >= total_steps:
                break

            input_ids = input_ids.to(device)
            targets = targets.to(device)

            lr = get_lr(global_step, args.warmup_steps, args.lr, total_steps)
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr

            student_x = student.drop(student.token_emb(input_ids))
            for block in student.blocks:
                student_x = block(student_x)
            student_hidden = student.final_norm(student_x)
            student_logits = student.lm_head(student_hidden)

            with torch.no_grad():
                teacher_x = teacher.drop(teacher.token_emb(input_ids))
                for block in teacher.blocks:
                    teacher_x = block(teacher_x)
                teacher_hidden = teacher.final_norm(teacher_x)
                teacher_logits = teacher.lm_head(teacher_hidden)

            total_loss, metrics = compute_distillation_loss(
                student_logits=student_logits,
                teacher_logits=teacher_logits,
                targets=targets,
                student_hidden=student_hidden,
                teacher_hidden=teacher_hidden,
                regressor=regressor,
                temperature=args.temp,
                alpha_ce=args.alpha_ce,
                alpha_kl=args.alpha_kl,
                alpha_mse=args.alpha_mse,
            )

            optimizer.zero_grad(set_to_none=True)
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(all_params, args.grad_clip)
            optimizer.step()

            global_step += 1

            if global_step % args.log_interval == 0:
                print(
                    f"[STEP {global_step}/{total_steps}] "
                    f"epoch={epoch} "
                    f"ce={metrics['ce_loss']:.4f} "
                    f"kl={metrics['kl_loss']:.4f} "
                    f"mse={metrics['mse_loss']:.4f} "
                    f"total={metrics['total_loss']:.4f} "
                    f"lr={lr:.2e}"
                )

            if global_step % args.save_interval == 0:
                save_checkpoint(
                    student, optimizer, global_step, epoch,
                    metrics["total_loss"], checkpoint_dir,
                )

    save_checkpoint(student, optimizer, global_step, args.epochs, 0.0, checkpoint_dir)

    regressor_path = checkpoint_dir / "regressor.pt"
    torch.save(regressor.state_dict(), regressor_path)

    print("[DONE] Knowledge distillation complete.")


if __name__ == "__main__":
    train(parse_args())
