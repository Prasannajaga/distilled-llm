#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import re
import sys
from pathlib import Path

from datasets import Dataset, load_dataset, load_from_disk

DEFAULT_BASE_MODEL = "/media/prasanna/716F26140AED9B67/models/models--Qwen--Qwen2-Math-1.5B-Instruct"
DEFAULT_DATASET_PATH = "/media/prasanna/716F26140AED9B67/datasets/GSM8K"
LOGGER = logging.getLogger("train_teacher_unsloth")


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
    p = argparse.ArgumentParser(description="Fine-tune teacher model on GSM8K using Unsloth LoRA")
    p.add_argument("--base-model", type=str, default=DEFAULT_BASE_MODEL)
    p.add_argument("--dataset-path", type=str, default=DEFAULT_DATASET_PATH)
    p.add_argument("--split", choices=["train", "test"], default="train")
    p.add_argument("--max-train-samples", type=int, default=0)
    p.add_argument("--max-eval-samples", type=int, default=256)
    p.add_argument("--train-jsonl", type=str, default=None, help="Optional JSONL with question/answer for training.")
    p.add_argument("--eval-jsonl", type=str, default=None, help="Optional JSONL with question/answer for eval.")
    p.add_argument("--output-dir", type=str, default="newoutput/teacher-gsm8k-unsloth-lora")
    p.add_argument("--merged-output-dir", type=str, default="newoutput/teacher-gsm8k-unsloth-merged")
    p.add_argument("--dataset-cache-root", type=str, default="newoutput/cache/teacher_unsloth")
    p.add_argument("--use-dataset-cache", type=int, choices=[0, 1], default=1)

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
        "--answer-style",
        choices=["cot_final_marker", "raw"],
        default="cot_final_marker",
        help="Target style for assistant answer text.",
    )

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
        LOGGER.info("DATA using local Arrow split: %s", split_arrow)
        return Dataset.from_file(str(split_arrow))
    except FileNotFoundError:
        LOGGER.info("DATA local Arrow not found, falling back to Hugging Face openai/gsm8k")
        return load_dataset("openai/gsm8k", "main", split=split)


def load_jsonl_dataset(path: str) -> Dataset:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"JSONL dataset not found: {p}")
    ds = load_dataset("json", data_files=str(p), split="train")
    if "question" not in ds.column_names or "answer" not in ds.column_names:
        raise ValueError(f"JSONL must contain 'question' and 'answer' fields: {p}")
    return ds


def _extract_final_number(answer: str) -> str | None:
    marker = re.search(r"####\s*([^\n\r]+)", answer)
    if marker:
        tail = marker.group(1)
        nums = re.findall(r"-?[0-9]+(?:\.[0-9]+)?", tail.replace(",", ""))
        if nums:
            return nums[-1]
    answer_marker = re.search(r"(?is)the answer is\s*(-?[0-9]+(?:\.[0-9]+)?)", answer)
    if answer_marker:
        return answer_marker.group(1)
    nums = re.findall(r"-?[0-9]+(?:\.[0-9]+)?", answer.replace(",", ""))
    return nums[-1] if nums else None


def _normalize_answer(answer: str, style: str) -> str:
    answer = answer.strip()
    if style == "raw":
        return answer

    rationale = answer
    if "####" in rationale:
        rationale = rationale.split("####", 1)[0].strip()
    rationale = re.sub(r"\s+", " ", rationale).strip()
    final_num = _extract_final_number(answer)
    marker = f"The answer is {final_num}." if final_num is not None else "The answer is [unknown]."
    if rationale:
        return f"{rationale}\n{marker}"
    return marker


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


def _format_row(question: str, answer: str, answer_style: str, tokenizer: object) -> str:
    question = question.strip()
    answer = _normalize_answer(answer, answer_style)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
        {"role": "assistant", "content": answer},
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
    return _legacy_chat_text(question, answer)


def prepare_dataset(ds: Dataset, answer_style: str, tokenizer: object) -> Dataset:
    def _mapper(row: dict) -> dict:
        q = str(row.get("question", "")).strip()
        a = str(row.get("answer", "")).strip()
        return {"text": _format_row(q, a, answer_style, tokenizer)}

    return ds.map(_mapper, remove_columns=ds.column_names)


def _stable_id(value: str | None) -> str:
    if value is None:
        return "none"
    text = value.strip()
    return text if text else "none"


def _dataset_cache_key(
    *,
    stage: str,
    resolved_base_model: str,
    dataset_path: str,
    split: str,
    train_jsonl: str | None,
    eval_jsonl: str | None,
    max_train_samples: int,
    max_eval_samples: int,
    answer_style: str,
    max_seq_length: int,
    chat_template_sig: str,
) -> str:
    payload = {
        "v": 1,
        "stage": stage,
        "resolved_base_model": _stable_id(resolved_base_model),
        "dataset_path": _stable_id(dataset_path),
        "split": split,
        "train_jsonl": _stable_id(train_jsonl),
        "eval_jsonl": _stable_id(eval_jsonl),
        "max_train_samples": int(max_train_samples),
        "max_eval_samples": int(max_eval_samples),
        "answer_style": answer_style,
        "max_seq_length": int(max_seq_length),
        "chat_template_sig": chat_template_sig,
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


def _load_or_build_prepared_dataset(
    *,
    cache_root: Path,
    cache_key: str,
    split_name: str,
    raw_ds: Dataset,
    answer_style: str,
    tokenizer: object,
    use_cache: bool,
) -> Dataset:
    cache_dir = cache_root / f"{split_name}_{cache_key}"
    if use_cache and cache_dir.exists():
        LOGGER.info("DATA loading prepared cache: %s", cache_dir)
        return load_from_disk(str(cache_dir))

    ds = prepare_dataset(raw_ds, answer_style, tokenizer)
    if use_cache:
        cache_dir.parent.mkdir(parents=True, exist_ok=True)
        LOGGER.info("DATA saving prepared cache: %s", cache_dir)
        ds.save_to_disk(str(cache_dir))
    return ds


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

        local_snapshot = snapshot_download(
            repo_id=model_ref,
            local_files_only=True,
        )
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
    LOGGER.info("CFG split=%s max_train_samples=%s", args.split, args.max_train_samples)

    train_raw = load_jsonl_dataset(args.train_jsonl) if args.train_jsonl else load_gsm8k(args.split, args.dataset_path)
    if args.max_train_samples > 0:
        train_raw = train_raw.select(range(min(args.max_train_samples, len(train_raw))))

    eval_raw = load_jsonl_dataset(args.eval_jsonl) if args.eval_jsonl else load_gsm8k("test", args.dataset_path)
    if args.max_eval_samples > 0:
        eval_raw = eval_raw.select(range(min(args.max_eval_samples, len(eval_raw))))

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=resolved_base_model,
        max_seq_length=args.max_seq_length,
        load_in_4bit=bool(args.load_in_4bit),
    )
    chat_template_sig = _chat_template_signature(tokenizer)
    LOGGER.info("CFG chat_template_sig=%s", chat_template_sig)

    cache_root = Path(args.dataset_cache_root)
    use_cache = bool(args.use_dataset_cache)
    train_cache_key = _dataset_cache_key(
        stage="train",
        resolved_base_model=resolved_base_model,
        dataset_path=args.dataset_path,
        split=args.split,
        train_jsonl=args.train_jsonl,
        eval_jsonl=args.eval_jsonl,
        max_train_samples=args.max_train_samples,
        max_eval_samples=args.max_eval_samples,
        answer_style=args.answer_style,
        max_seq_length=args.max_seq_length,
        chat_template_sig=chat_template_sig,
    )
    eval_cache_key = _dataset_cache_key(
        stage="eval",
        resolved_base_model=resolved_base_model,
        dataset_path=args.dataset_path,
        split="test",
        train_jsonl=args.train_jsonl,
        eval_jsonl=args.eval_jsonl,
        max_train_samples=args.max_train_samples,
        max_eval_samples=args.max_eval_samples,
        answer_style=args.answer_style,
        max_seq_length=args.max_seq_length,
        chat_template_sig=chat_template_sig,
    )
    train_ds = _load_or_build_prepared_dataset(
        cache_root=cache_root,
        cache_key=train_cache_key,
        split_name="train",
        raw_ds=train_raw,
        answer_style=args.answer_style,
        tokenizer=tokenizer,
        use_cache=use_cache,
    )
    eval_ds = _load_or_build_prepared_dataset(
        cache_root=cache_root,
        cache_key=eval_cache_key,
        split_name="eval",
        raw_ds=eval_raw,
        answer_style=args.answer_style,
        tokenizer=tokenizer,
        use_cache=use_cache,
    )
    LOGGER.info("DATA train=%s eval=%s cache=%s", f"{len(train_ds):,}", f"{len(eval_ds):,}", use_cache)

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

    LOGGER.info("TRAIN starting Unsloth LoRA fine-tune...")
    trainer.train()

    LOGGER.info("SAVE LoRA adapters to %s", out_dir)
    model.save_pretrained(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))

    if args.merge_16bit:
        merged_dir = Path(args.merged_output_dir)
        merged_dir.mkdir(parents=True, exist_ok=True)
        LOGGER.info("SAVE merged 16-bit model to %s", merged_dir)
        model.save_pretrained_merged(str(merged_dir), tokenizer, save_method="merged_16bit")
        merged_files = list(merged_dir.glob("*"))
        if not merged_files:
            raise RuntimeError(
                "Merged model export produced no files. "
                "This usually happens when the base model cannot be resolved locally. "
                "Try passing a local model path with --base-model."
            )

    LOGGER.info("DONE teacher fine-tune complete.")


if __name__ == "__main__":
    main()
