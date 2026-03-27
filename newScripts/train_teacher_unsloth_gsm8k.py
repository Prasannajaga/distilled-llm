#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from pathlib import Path

from datasets import Dataset, load_dataset

DEFAULT_BASE_MODEL = "Qwen/Qwen2-Math-1.5B-Instruct"
DEFAULT_DATASET_PATH = "/media/prasanna/716F26140AED9B67/datasets/GSM8K"


SYSTEM_PROMPT = (
    "You are a careful math tutor. Solve step by step and end with "
    "'The answer is <number>.'."
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fine-tune teacher model on GSM8K using Unsloth LoRA")
    p.add_argument("--base-model", type=str, default=DEFAULT_BASE_MODEL)
    p.add_argument("--dataset-path", type=str, default=DEFAULT_DATASET_PATH)
    p.add_argument("--split", choices=["train", "test"], default="train")
    p.add_argument("--max-train-samples", type=int, default=0)
    p.add_argument("--max-eval-samples", type=int, default=256)
    p.add_argument("--output-dir", type=str, default="newoutput/teacher-gsm8k-unsloth-lora")
    p.add_argument("--merged-output-dir", type=str, default="newoutput/teacher-gsm8k-unsloth-merged")

    p.add_argument("--max-seq-length", type=int, default=2048)
    p.add_argument("--load-in-4bit", type=int, choices=[0, 1], default=1)

    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=16)
    p.add_argument("--lora-dropout", type=float, default=0.0)

    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--epochs", type=float, default=2.0)
    p.add_argument("--learning-rate", type=float, default=2e-4)
    p.add_argument("--warmup-ratio", type=float, default=0.05)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--logging-steps", type=int, default=10)
    p.add_argument("--save-steps", type=int, default=1000)
    p.add_argument("--seed", type=int, default=42)

    p.add_argument(
        "--merge-16bit",
        type=int,
        choices=[0, 1],
        default=1,
        help="If 1, saves merged 16-bit model for regular HF/vLLM inference.",
    )
    return p.parse_args()


def _search_arrow(root: Path, split: str) -> Path | None:
    if not root.exists() or not root.is_dir():
        return None
    target_name = f"gsm8k-{split}.arrow"
    direct = root / target_name
    if direct.exists():
        return direct
    matches = sorted(root.rglob(target_name))
    return matches[0] if matches else None


def resolve_split_arrow(dataset_path: str, split: str) -> Path:
    requested = Path(dataset_path)
    target_name = f"gsm8k-{split}.arrow"

    if requested.exists():
        if requested.is_file():
            if requested.name != target_name:
                raise FileNotFoundError(
                    f"Expected file name {target_name} for split '{split}', got: {requested}"
                )
            return requested
        found = _search_arrow(requested, split)
        if found is not None:
            return found

    fallback_roots = [
        Path("/media/prasanna/716F26140AED9B67/datasets/openai___gsm8k"),
        Path("/media/prasanna/716F26140AED9B67/datasets/openai___gsm8k/main"),
    ]
    if requested.name.lower() == "gsm8k":
        fallback_roots = [requested.parent / "openai___gsm8k", requested.parent / "openai___gsm8k" / "main"] + fallback_roots

    for root in fallback_roots:
        found = _search_arrow(root, split)
        if found is not None:
            return found

    raise FileNotFoundError(
        f"Could not locate {target_name}. Checked {requested} and standard openai___gsm8k fallbacks."
    )


def load_gsm8k(split: str, dataset_path: str) -> Dataset:
    try:
        split_arrow = resolve_split_arrow(dataset_path, split)
        print(f"[DATA] Using local Arrow split: {split_arrow}")
        return Dataset.from_file(str(split_arrow))
    except FileNotFoundError:
        print("[DATA] Local Arrow not found, falling back to Hugging Face openai/gsm8k")
        return load_dataset("openai/gsm8k", "main", split=split)


def _format_row(question: str, answer: str) -> str:
    question = question.strip()
    answer = answer.strip()
    return (
        f"<|system|>\\n{SYSTEM_PROMPT}\\n"
        f"<|user|>\\n{question}\\n"
        f"<|assistant|>\\n{answer}"
    )


def prepare_dataset(ds: Dataset) -> Dataset:
    def _mapper(row: dict) -> dict:
        q = str(row.get("question", "")).strip()
        a = str(row.get("answer", "")).strip()
        return {"text": _format_row(q, a)}

    return ds.map(_mapper, remove_columns=ds.column_names)


def resolve_base_model_ref(model_ref: str) -> str:
    local_path = Path(model_ref).expanduser()
    if local_path.exists():
        return str(local_path.resolve())
    try:
        from huggingface_hub import snapshot_download

        local_snapshot = snapshot_download(
            repo_id=model_ref,
            local_files_only=True,
        )
        return str(Path(local_snapshot).resolve())
    except Exception:
        return model_ref


def main() -> None:
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
    print(f"[CFG] base_model={args.base_model}")
    if resolved_base_model != args.base_model:
        print(f"[CFG] resolved_base_model={resolved_base_model}")
    print(f"[CFG] split={args.split} max_train_samples={args.max_train_samples}")

    train_raw = load_gsm8k(args.split, args.dataset_path)
    if args.max_train_samples > 0:
        train_raw = train_raw.select(range(min(args.max_train_samples, len(train_raw))))

    eval_raw = load_gsm8k("test", args.dataset_path)
    if args.max_eval_samples > 0:
        eval_raw = eval_raw.select(range(min(args.max_eval_samples, len(eval_raw))))

    train_ds = prepare_dataset(train_raw)
    eval_ds = prepare_dataset(eval_raw)
    print(f"[DATA] train={len(train_ds):,} eval={len(eval_ds):,}")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=resolved_base_model,
        max_seq_length=args.max_seq_length,
        load_in_4bit=bool(args.load_in_4bit),
    )

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

    print("[TRAIN] Starting Unsloth LoRA fine-tune...")
    trainer.train()

    print(f"[SAVE] Saving LoRA adapters to {out_dir}")
    model.save_pretrained(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))

    if args.merge_16bit:
        merged_dir = Path(args.merged_output_dir)
        merged_dir.mkdir(parents=True, exist_ok=True)
        print(f"[SAVE] Saving merged 16-bit model to {merged_dir}")
        model.save_pretrained_merged(str(merged_dir), tokenizer, save_method="merged_16bit")
        merged_files = list(merged_dir.glob("*"))
        if not merged_files:
            raise RuntimeError(
                "Merged model export produced no files. "
                "This usually happens when the base model cannot be resolved locally. "
                "Try passing a local model path with --base-model."
            )

    print("[DONE] Teacher fine-tune complete.")


if __name__ == "__main__":
    main()
