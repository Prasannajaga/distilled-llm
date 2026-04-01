#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import logging
import os
import random
import sys
from pathlib import Path

from datasets import Dataset, load_dataset

DEFAULT_BASE_MODEL = "/media/prasanna/716F26140AED9B67/models/models--Qwen--Qwen2-0.5B-Instruct"
LOGGER = logging.getLogger("train_student_unsloth")

SYSTEM_PROMPT = (
    "You are a careful math tutor. Solve step by step and end with "
    "'The answer is <number>.'."
)


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
        force=True,
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Distill student model using teacher-generated GSM8K-style JSONL.")
    p.add_argument("--base-model", type=str, default=DEFAULT_BASE_MODEL)
    p.add_argument("--train-jsonl", type=str, required=True, help="Required JSONL with question/answer fields.")
    p.add_argument("--eval-jsonl", type=str, default=None, help="Optional eval JSONL with question/answer fields.")
    p.add_argument("--max-train-samples", type=int, default=0)
    p.add_argument("--max-eval-samples", type=int, default=512)
    p.add_argument(
        "--val-split-ratio",
        type=float,
        default=0.0,
        help="If >0 and --eval-jsonl is not provided, carve eval split from train JSONL.",
    )
    p.add_argument(
        "--split-seed",
        type=int,
        default=42,
        help="Seed used when creating deterministic train/eval split from train JSONL.",
    )
    p.add_argument("--output-dir", type=str, default="output/student-gsm8k-distill-lora")
    p.add_argument("--merged-output-dir", type=str, default="output/student-gsm8k-distill-merged")

    p.add_argument("--max-seq-length", type=int, default=2048)
    p.add_argument("--load-in-4bit", type=int, choices=[0, 1], default=1)

    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=16)
    p.add_argument("--lora-dropout", type=float, default=0.0)

    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--epochs", type=float, default=1.0)
    p.add_argument("--learning-rate", type=float, default=1e-4)
    p.add_argument("--warmup-ratio", type=float, default=0.05)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--logging-steps", type=int, default=10)
    p.add_argument("--save-steps", type=int, default=1000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--merge-16bit", type=int, choices=[0, 1], default=1)
    return p.parse_args()


def load_jsonl_dataset(path: str) -> Dataset:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"JSONL dataset not found: {p}")
    ds = load_dataset("json", data_files=str(p), split="train")
    if "question" not in ds.column_names or "answer" not in ds.column_names:
        raise ValueError(f"JSONL must contain 'question' and 'answer' fields: {p}")
    return ds


def _legacy_chat_text(question: str, answer: str) -> str:
    return (
        f"<|system|>\\n{SYSTEM_PROMPT}\\n"
        f"<|user|>\\n{question}\\n"
        f"<|assistant|>\\n{answer}"
    )


def _chat_template_signature(tokenizer: object) -> str:
    tmpl = getattr(tokenizer, "chat_template", None)
    if isinstance(tmpl, str) and tmpl.strip():
        return hashlib.sha1(tmpl.encode("utf-8")).hexdigest()[:12]
    return "no-template"


def _format_row(question: str, answer: str, tokenizer: object) -> str:
    q = question.strip()
    a = answer.strip()
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": q},
        {"role": "assistant", "content": a},
    ]
    apply_fn = getattr(tokenizer, "apply_chat_template", None)
    if callable(apply_fn):
        try:
            rendered = apply_fn(
                messages,
                tokenize=False,
                add_generation_prompt=False,
            )
            if isinstance(rendered, str) and rendered.strip():
                return rendered
        except Exception as exc:
            LOGGER.warning("CHAT template apply failed, using legacy fallback: %s", exc)
    return _legacy_chat_text(q, a)


def prepare_dataset(ds: Dataset, tokenizer: object) -> Dataset:
    def _mapper(row: dict) -> dict:
        q = str(row.get("question", "")).strip()
        a = str(row.get("answer", "")).strip()
        return {"text": _format_row(q, a, tokenizer)}

    return ds.map(_mapper, remove_columns=ds.column_names)


def split_train_eval_dataset(ds: Dataset, val_split_ratio: float, seed: int) -> tuple[Dataset, Dataset]:
    if len(ds) < 2:
        raise ValueError("Need at least 2 rows to build train/eval split.")
    ratio = float(val_split_ratio)
    if ratio <= 0.0 or ratio >= 1.0:
        raise ValueError("--val-split-ratio must be in (0, 1).")
    total = len(ds)
    eval_size = int(round(total * ratio))
    eval_size = max(1, min(total - 1, eval_size))
    indices = list(range(total))
    rnd = random.Random(int(seed))
    rnd.shuffle(indices)
    eval_idx = sorted(indices[:eval_size])
    train_idx = sorted(indices[eval_size:])
    return ds.select(train_idx), ds.select(eval_idx)


def resolve_base_model_ref(model_ref: str) -> str:
    local_path = Path(model_ref).expanduser()
    if local_path.exists():
        p = local_path.resolve()
        # Handle HF cache-root layout:
        #   models--ORG--NAME/{refs,snapshots/<rev>/...}
        if p.is_dir() and (p / "snapshots").is_dir() and (p / "refs").is_dir():
            snap_root = p / "snapshots"
            chosen: Path | None = None
            ref_main = p / "refs" / "main"
            if ref_main.exists():
                rev = ref_main.read_text(encoding="utf-8").strip()
                cand = snap_root / rev
                if cand.is_dir():
                    chosen = cand
            if chosen is None:
                snaps = sorted(
                    [d for d in snap_root.iterdir() if d.is_dir()],
                    key=lambda d: d.stat().st_mtime,
                    reverse=True,
                )
                if snaps:
                    chosen = snaps[0]
            if chosen is not None:
                p = chosen
        return str(p)
    try:
        from huggingface_hub import snapshot_download

        local_snapshot = snapshot_download(repo_id=model_ref, local_files_only=True)
        return str(Path(local_snapshot).resolve())
    except Exception:
        return model_ref


def main() -> None:
    configure_logging()
    args = parse_args()

    try:
        from unsloth import FastLanguageModel
        from trl import SFTConfig, SFTTrainer
    except Exception as exc:
        raise RuntimeError(
            "Missing training deps. Install with:\n"
            "  uv pip install unsloth trl peft bitsandbytes huggingface_hub\n"
            f"Original import error: {exc}"
        ) from exc

    resolved_base_model = resolve_base_model_ref(args.base_model)
    LOGGER.info("CFG base_model=%s", args.base_model)
    if resolved_base_model != args.base_model:
        LOGGER.info("CFG resolved_base_model=%s", resolved_base_model)

    train_raw = load_jsonl_dataset(args.train_jsonl)
    if args.max_train_samples > 0:
        train_raw = train_raw.select(range(min(args.max_train_samples, len(train_raw))))
    if args.eval_jsonl:
        eval_raw = load_jsonl_dataset(args.eval_jsonl)
    elif float(args.val_split_ratio) > 0.0:
        train_raw, eval_raw = split_train_eval_dataset(
            train_raw,
            val_split_ratio=float(args.val_split_ratio),
            seed=int(args.split_seed),
        )
    else:
        eval_raw = train_raw
    if args.max_eval_samples > 0:
        eval_raw = eval_raw.select(range(min(args.max_eval_samples, len(eval_raw))))

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=resolved_base_model,
        max_seq_length=args.max_seq_length,
        load_in_4bit=bool(args.load_in_4bit),
    )
    chat_template_sig = _chat_template_signature(tokenizer)
    LOGGER.info("CFG chat_template_sig=%s", chat_template_sig)

    train_ds = prepare_dataset(train_raw, tokenizer)
    eval_ds = prepare_dataset(eval_raw, tokenizer)
    LOGGER.info("DATA train=%s eval=%s", f"{len(train_ds):,}", f"{len(eval_ds):,}")

    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        use_gradient_checkpointing="unsloth",
        random_state=args.seed,
    )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    bf16_ok = os.environ.get("UNSLOTH_FORCE_FP16", "0") != "1"
    sft_config = SFTConfig(
        output_dir=str(out_dir),
        dataset_text_field="text",
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        lr_scheduler_type="cosine",
        optim="adamw_8bit",
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        bf16=bf16_ok,
        fp16=not bf16_ok,
        seed=args.seed,
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        args=sft_config,
    )

    LOGGER.info("TRAIN starting student distillation fine-tune...")
    trainer.train()

    LOGGER.info("SAVE lora_dir=%s", out_dir)
    model.save_pretrained(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))

    if args.merge_16bit:
        merged_dir = Path(args.merged_output_dir)
        merged_dir.mkdir(parents=True, exist_ok=True)
        LOGGER.info("SAVE merged_dir=%s", merged_dir)
        model.save_pretrained_merged(str(merged_dir), tokenizer, save_method="merged_16bit")
        if not list(merged_dir.glob("*")):
            raise RuntimeError("Merged student export produced no files.")

    LOGGER.info("DONE student distillation complete.")


if __name__ == "__main__":
    main()
