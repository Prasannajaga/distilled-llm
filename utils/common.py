from __future__ import annotations

import json
import math
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn as nn


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


def params(model: nn.Module) -> float:
    """Return total number of parameters in millions."""
    return f"{sum(p.numel() for p in model.parameters()) / 1e6} M"

def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    checkpoint_dir: Path,
    config: Any,
    prefix: str = "ckpt",
    tokenizer: Any = None,
    scaler: Any = None,
    save_optimizer_state: bool = True,
) -> str:
    """Save a training checkpoint with optional config, tokenizer, and scaler state.

    Directory layout:
        checkpoint_dir/prefix/config.json       (written once)
        checkpoint_dir/prefix/tokenizer/         (written once)
        checkpoint_dir/prefix/prefix_{step}.pt   (written every call)
    """
    if not checkpoint_dir:
        raise ValueError("Checkpoint directory not configured.")

    prefix_dir = checkpoint_dir / prefix
    prefix_dir.mkdir(parents=True, exist_ok=True)

    config_path = prefix_dir / "config.json"
    if not config_path.exists():
        config_data = asdict(config)
        with open(config_path, "w") as f:
            json.dump(config_data, f, indent=2)
        print(f"[CHECKPOINT] Saved config: {config_path}")

    fname = f"{prefix}_{step}.pt"
    path = prefix_dir / fname
    payload: dict[str, Any] = {
        "model_state_dict": model.state_dict(),
        "training_step": step,
    }

    if tokenizer is not None:
        tokenizer_path = prefix_dir / "tokenizer.json"
        if not tokenizer_path.exists():
            try:
                tokenizer.save(tokenizer_path)
                print(f"[CHECKPOINT] Saved tokenizer: {tokenizer_path}")
            except Exception as e:
                print(f"[CHECKPOINT] Warning: Failed to save tokenizer: {e}")

    if save_optimizer_state:
        payload["optimizer_state_dict"] = optimizer.state_dict()

    if scaler is not None:
        try:
            payload["scaler_state_dict"] = scaler.state_dict()
        except Exception:
            payload["scaler_state_dict"] = None

    torch.save(payload, path)
    print(f"[CHECKPOINT] Saved checkpoint: {path}")
    return str(path)


def load_checkpoint(
    checkpoint_dir: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    prefix: str = "ckpt",
) -> tuple[int, int]:
    """Load the latest checkpoint from a prefix subdirectory.

    Scans checkpoint_dir/prefix/ for files matching prefix_{step}.pt and loads
    the one with the highest step number. Returns (step, 0) on success or
    (0, 0) when no checkpoint is found.
    """
    prefix_dir = checkpoint_dir / prefix
    if not prefix_dir.exists():
        print("[INIT] No checkpoint found. Starting from scratch.")
        return 0, 0

    candidates = sorted(prefix_dir.glob(f"{prefix}_*.pt"))
    if not candidates:
        print("[INIT] No checkpoint found. Starting from scratch.")
        return 0, 0

    latest = candidates[-1]
    print(f"[RESUME] Loading checkpoint from {latest}")
    checkpoint = torch.load(latest, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    if "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    step = checkpoint.get("training_step", checkpoint.get("step", 0))
    return step, 0
