from __future__ import annotations

import json
import os
import platform
import tempfile
from dataclasses import asdict
from datetime import datetime, timezone
from typing import Any

import torch

from utils.config import TrainingConfig

CONFIG_SCHEMA_VERSION = "1.0"


def _get_token_id(tokenizer: Any, *attrs: str) -> int | None:
    for name in attrs:
        value = getattr(tokenizer, name, None)
        if value is not None:
            return int(value)
    return None


def _to_json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _to_json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_json_safe(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def build_model_config(
    training_cfg: TrainingConfig,
    tokenizer: Any = None,
    model: Any = None,
) -> dict[str, Any]:
    if model is not None:
        architecture = model.__class__.__name__
    else:
        architecture = "GQATransformer"

    try:
        import transformers

        transformers_version = transformers.__version__
    except Exception:
        transformers_version = None

    vocab_size = None
    if tokenizer is not None:
        try:
            vocab_size = int(len(tokenizer))
        except Exception:
            vocab_size = None

    payload: dict[str, Any] = {
        "architectures": [architecture],
        "model_type": "gqa_transformer",
        "hidden_size": int(training_cfg.n_embd),
        "num_hidden_layers": int(training_cfg.n_layer),
        "num_attention_heads": int(training_cfg.n_head),
        "num_key_value_heads": max(1, int(training_cfg.n_head) // 2),
        "max_position_embeddings": int(training_cfg.block_size),
        "vocab_size": vocab_size,
        "bos_token_id": _get_token_id(tokenizer, "bos_token_id", "bos_id"),
        "eos_token_id": _get_token_id(tokenizer, "eos_token_id", "eos_id"),
        "pad_token_id": _get_token_id(tokenizer, "pad_token_id", "pad_id"),
        "torch_dtype": str(getattr(training_cfg, "amp_dtype", "float32")).replace("fp", "float"),
        "tie_word_embeddings": False,
        "use_cache": True,
        "transformers_version": transformers_version,
    }
    return {k: v for k, v in payload.items() if v is not None}


def build_checkpoint_config_payload(
    training_cfg: TrainingConfig,
    tokenizer: Any = None,
    model: Any = None,
) -> dict[str, Any]:
    return {
        "schema_version": CONFIG_SCHEMA_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "library": {
            "project": "distilled-llm",
            "python_version": platform.python_version(),
            "torch_version": torch.__version__,
        },
        "model_config": build_model_config(training_cfg, tokenizer=tokenizer, model=model),
        "training_config": _to_json_safe(asdict(training_cfg)),
    }


def save_config_json(
    config_path: str,
    training_cfg: TrainingConfig,
    tokenizer: Any = None,
    model: Any = None,
) -> None:
    payload = build_checkpoint_config_payload(
        training_cfg=training_cfg,
        tokenizer=tokenizer,
        model=model,
    )
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    fd, temp_path = tempfile.mkstemp(suffix=".json", dir=os.path.dirname(config_path))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, sort_keys=True)
        os.replace(temp_path, config_path)
    except Exception:
        if os.path.exists(temp_path):
            os.unlink(temp_path)
        raise
