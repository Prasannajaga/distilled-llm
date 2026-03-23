from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import random
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from datasets import Dataset
from huggingface_hub import snapshot_download
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

DEFAULT_STUDENT_MODEL = "Qwen/Qwen2-0.5B-Instruct"
DEFAULT_TEACHER_MODEL = "Qwen/Qwen2-Math-1.5B-Instruct"
DEFAULT_GSM8K_PATH = "/media/prasanna/716F26140AED9B67/datasets/GSM8K"
LOGGER = logging.getLogger("eval_qwen_gsm8k")


@dataclass
class EvalSample:
    sample_id: int
    question: str
    answer: str
    raw_answer: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate teacher and student Hugging Face models on local GSM8K and compare results."
    )
    parser.add_argument("--student-model", type=str, default=DEFAULT_STUDENT_MODEL)
    parser.add_argument("--teacher-model", type=str, default=DEFAULT_TEACHER_MODEL)
    parser.add_argument(
        "--dataset-path",
        type=str,
        default=DEFAULT_GSM8K_PATH,
        help="Path hint to GSM8K dataset directory or split arrow file.",
    )
    parser.add_argument("--split", type=str, choices=["train", "test"], default="test")
    parser.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help="Maximum number of samples to evaluate. Set <=0 to use full split.",
    )
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--device", type=str, choices=["auto", "cpu", "cuda", "mps"], default="auto")
    parser.add_argument(
        "--dtype",
        type=str,
        choices=["auto", "float32", "bfloat16", "float16"],
        default="auto",
    )

    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument("--numeric-tolerance", type=float, default=1e-3)

    parser.add_argument(
        "--render-mode",
        type=str,
        choices=["auto", "chat", "plain"],
        default="auto",
    )
    parser.add_argument(
        "--prompt-template",
        type=str,
        default="Question: {question}\nAnswer:",
        help="Used only in plain render mode. Must contain {question}.",
    )
    parser.add_argument(
        "--system-prompt",
        type=str,
        default="You are a math assistant. Solve carefully and end with 'Final answer: <number>'.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory. Default: newoutput/evals/beforeDistillation.<timestamp>",
    )
    parser.add_argument("--download-retries", type=int, default=20)
    parser.add_argument("--download-retry-wait-sec", type=int, default=15)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--debug", action="store_true", help="Enable verbose stage logging.")
    parser.add_argument(
        "--log-interval-samples",
        type=int,
        default=25,
        help="Emit per-sample progress log every N samples.",
    )
    parser.add_argument(
        "--oom-fallback-to-cpu",
        action="store_true",
        default=True,
        help="On CUDA OOM, move current model to CPU and continue.",
    )
    parser.add_argument(
        "--oom-retry-halves",
        type=int,
        default=3,
        help="Number of retries that halve max_new_tokens after CUDA OOM.",
    )
    return parser.parse_args()


def configure_logging(debug: bool) -> None:
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,
    )


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def normalize_text(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"\\boxed\{([^{}]+)\}", r"\1", text)
    text = text.replace(",", "")
    text = re.sub(r"[^a-z0-9\.\-\/ ]+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def parse_number(text: str) -> float | None:
    if not text:
        return None

    candidate = text.strip()
    candidate = re.sub(r"\\boxed\{([^{}]+)\}", r"\1", candidate)
    candidate = candidate.replace(",", "")
    candidate = candidate.replace("−", "-")

    frac = re.search(r"\\frac\{([\-0-9\.]+)\}\{([\-0-9\.]+)\}", candidate)
    if frac:
        num = float(frac.group(1))
        den = float(frac.group(2))
        if den != 0:
            return num / den

    frac2 = re.search(r"([\-]?[0-9]+(?:\.[0-9]+)?)\s*/\s*([\-]?[0-9]+(?:\.[0-9]+)?)", candidate)
    if frac2:
        num = float(frac2.group(1))
        den = float(frac2.group(2))
        if den != 0:
            return num / den

    numbers = re.findall(r"[\-]?[0-9]+(?:\.[0-9]+)?", candidate)
    if not numbers:
        return None
    try:
        return float(numbers[-1])
    except ValueError:
        return None


def extract_final_answer(text: str) -> str:
    s = text.strip()
    if not s:
        return ""

    if "<|answer|>" in s:
        tail = s.split("<|answer|>")[-1]
        tail = tail.split("<|eos|>")[0]
        tail = tail.strip()
        if tail:
            return tail

    marker = re.search(r"####\s*([^\n\r]+)", s)
    if marker:
        return marker.group(1).strip()

    boxed = re.search(r"\\boxed\{([^{}]+)\}", s)
    if boxed:
        return boxed.group(1).strip()

    patterns = [
        r"(?is)final answer\s*[:=]\s*(.+)",
        r"(?is)\banswer\s*[:=]\s*(.+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, s)
        if match:
            candidate = match.group(1).strip().splitlines()[0].strip()
            if candidate:
                return candidate

    lines = [line.strip() for line in s.splitlines() if line.strip()]
    return lines[-1] if lines else s


def extract_gsm8k_gold_answer(raw_answer: str) -> str:
    raw = raw_answer.strip()
    marker = re.search(r"####\s*([^\n\r]+)", raw)
    if marker:
        return marker.group(1).strip()
    return extract_final_answer(raw)


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
    LOGGER.info("Resolving dataset split file for split='%s' from '%s'", split, requested)

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

    fallback_roots: list[Path] = []
    if requested.name.lower() == "gsm8k":
        fallback_roots.extend(
            [
                requested.parent / "openai___gsm8k",
                requested.parent / "openai--gsm8k",
                requested.parent / "openai_gsm8k",
                requested.parent / "openai___gsm8k" / "main",
            ]
        )
    fallback_roots.extend(
        [
            Path("/media/prasanna/716F26140AED9B67/datasets/openai___gsm8k"),
            Path("/media/prasanna/716F26140AED9B67/datasets/openai___gsm8k/main"),
        ]
    )

    for root in fallback_roots:
        found = _search_arrow(root, split)
        if found is not None:
            return found

    raise FileNotFoundError(
        "Could not locate GSM8K split arrow file. "
        f"Checked requested path '{requested}' and common fallbacks for '{target_name}'."
    )


def load_gsm8k_samples(dataset_path: str, split: str, max_samples: int) -> tuple[list[EvalSample], str]:
    split_arrow = resolve_split_arrow(dataset_path, split)
    LOGGER.info("Loading dataset from %s", split_arrow)
    ds = Dataset.from_file(str(split_arrow))
    LOGGER.info("Loaded arrow with %d rows", len(ds))

    samples: list[EvalSample] = []
    max_count = max_samples if max_samples and max_samples > 0 else None

    for idx, row in enumerate(ds):
        if not isinstance(row, dict):
            continue
        question = str(row.get("question", "")).strip()
        raw_answer = str(row.get("answer", "")).strip()
        if not question or not raw_answer:
            continue
        answer = extract_gsm8k_gold_answer(raw_answer)
        if not answer:
            continue

        samples.append(
            EvalSample(
                sample_id=idx,
                question=question,
                answer=answer,
                raw_answer=raw_answer,
            )
        )
        if max_count is not None and len(samples) >= max_count:
            break

    if not samples:
        raise ValueError("No valid GSM8K rows found with question/answer values.")
    return samples, str(split_arrow)


def get_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device_arg)


def get_dtype(dtype_arg: str, device: torch.device) -> torch.dtype:
    mapping: dict[str, torch.dtype] = {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }
    if dtype_arg != "auto":
        requested = mapping[dtype_arg]
        if device.type == "cpu" and requested in {torch.float16, torch.bfloat16}:
            return torch.float32
        return requested

    if device.type == "cuda":
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    if device.type == "mps":
        return torch.float16
    return torch.float32


def build_inputs(
    *,
    tokenizer: Any,
    question: str,
    device: torch.device,
    render_mode: str,
    prompt_template: str,
    system_prompt: str,
) -> tuple[dict[str, torch.Tensor], str]:
    if render_mode in {"auto", "chat"} and hasattr(tokenizer, "apply_chat_template"):
        messages: list[dict[str, str]] = []
        if system_prompt.strip():
            messages.append({"role": "system", "content": system_prompt.strip()})
        messages.append({"role": "user", "content": question})
        try:
            input_ids = tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
            )
            if not isinstance(input_ids, torch.Tensor):
                input_ids = torch.as_tensor(input_ids)
            if input_ids.dim() == 1:
                input_ids = input_ids.unsqueeze(0)
            input_ids = input_ids.to(device)
            attention_mask = torch.ones_like(input_ids, dtype=torch.long, device=device)
            return {"input_ids": input_ids, "attention_mask": attention_mask}, "chat"
        except Exception:
            if render_mode == "chat":
                raise

    prompt = prompt_template.format(question=question)
    encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded.get("attention_mask")
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids, dtype=torch.long)
    attention_mask = attention_mask.to(device)
    return {"input_ids": input_ids, "attention_mask": attention_mask}, "plain"


def resolve_model_snapshot(
    *,
    model_id: str,
    download_retries: int,
    download_retry_wait_sec: int,
    local_files_only: bool,
) -> str:
    attempts = max(1, int(download_retries))
    wait_s = max(1, int(download_retry_wait_sec))
    last_online_error: Exception | None = None
    LOGGER.info(
        "Resolving model snapshot: model_id=%s local_files_only=%s retries=%d wait_s=%d",
        model_id,
        local_files_only,
        attempts,
        wait_s,
    )

    if local_files_only:
        LOGGER.info("Using local-files-only mode for %s", model_id)
        return snapshot_download(
            repo_id=model_id,
            repo_type="model",
            local_files_only=True,
        )

    for attempt in range(1, attempts + 1):
        try:
            LOGGER.info("Snapshot download attempt %d/%d for %s", attempt, attempts, model_id)
            return snapshot_download(
                repo_id=model_id,
                repo_type="model",
                cache_dir="/media/prasanna/716F26140AED9B67/models", 
            )
        except Exception as online_err:  # noqa: BLE001
            last_online_error = online_err
            if attempt >= attempts:
                break
            LOGGER.warning(
                "Download failed for %s (attempt %d/%d): %s: %s",
                model_id,
                attempt,
                attempts,
                type(online_err).__name__,
                online_err,
            )
            LOGGER.warning("Retrying in %ds...", wait_s)
            time.sleep(wait_s)

    assert last_online_error is not None
    try:
        local_source = snapshot_download(
            repo_id=model_id,
            repo_type="model",
            local_files_only=True,
        )
        LOGGER.warning(
            "Falling back to local cache for '%s' at %s. Online error: %s: %s",
            model_id,
            local_source,
            type(last_online_error).__name__,
            last_online_error,
        )
        return local_source
    except Exception as cache_err:  # noqa: BLE001
        raise RuntimeError(
            f"Unable to resolve model '{model_id}' after {attempts} download attempts.\n"
            f"Last online error: {type(last_online_error).__name__}: {last_online_error}\n"
            f"Local cache error: {type(cache_err).__name__}: {cache_err}"
        ) from last_online_error


def load_hf_model(
    model_id: str,
    *,
    device: torch.device,
    dtype: torch.dtype,
    download_retries: int,
    download_retry_wait_sec: int,
    local_files_only: bool,
) -> tuple[Any, Any]:
    LOGGER.info("Loading model '%s'...", model_id)
    local_source = resolve_model_snapshot(
        model_id=model_id,
        download_retries=download_retries,
        download_retry_wait_sec=download_retry_wait_sec,
        local_files_only=local_files_only,
    )
    local_source = os.path.abspath(local_source)
    LOGGER.info("Resolved model snapshot to %s", local_source)
    tokenizer = AutoTokenizer.from_pretrained(
        local_source,
        trust_remote_code=True,
        local_files_only=True,
    )
    LOGGER.info("Tokenizer loaded for %s", model_id)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    model_dir = Path(local_source)
    has_weights = any(
        (model_dir / name).exists()
        for name in (
            "model.safetensors",
            "pytorch_model.bin",
            "model.safetensors.index.json",
            "pytorch_model.bin.index.json",
        )
    )
    if not has_weights:
        raise RuntimeError(
            f"Model cache for '{model_id}' is incomplete at '{local_source}'. "
            "No weight file found yet. Re-run with network enabled to resume download."
        )

    LOGGER.info("Model weight files detected for %s", model_id)
    model = AutoModelForCausalLM.from_pretrained(
        local_source,
        dtype=dtype,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        local_files_only=True,
    )
    try:
        model.to(device)
    except torch.OutOfMemoryError as oom:
        if device.type != "cuda":
            raise
        LOGGER.warning("CUDA OOM while moving model '%s' to GPU: %s", model_id, oom)
        LOGGER.warning("Falling back to CPU for model '%s' during load.", model_id)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        model.to(torch.device("cpu"))
        device = torch.device("cpu")
    model.eval()
    LOGGER.info("Model loaded and moved to %s for %s", device, model_id)
    return model, tokenizer


def evaluate_one_model(
    *,
    label: str,
    model_id: str,
    samples: list[EvalSample],
    args: argparse.Namespace,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    LOGGER.info("Starting eval for label=%s model=%s", label, model_id)
    model, tokenizer = load_hf_model(
        model_id,
        device=device,
        dtype=dtype,
        download_retries=args.download_retries,
        download_retry_wait_sec=args.download_retry_wait_sec,
        local_files_only=args.local_files_only,
    )

    total = len(samples)
    LOGGER.info("Total samples for %s: %d", label, total)
    normalized_exact_count = 0
    numeric_total = 0
    numeric_tol_count = 0
    numeric_abs_error_sum = 0.0
    numeric_abs_error_count = 0
    total_generated_tokens = 0
    total_latency_s = 0.0
    truncation_hits = 0
    empty_prediction_count = 0
    chat_render_count = 0
    plain_render_count = 0

    records: list[dict[str, Any]] = []

    do_sample = float(args.temperature) > 0.0
    eos_token_id = tokenizer.eos_token_id
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else eos_token_id

    active_device = next(model.parameters()).device
    use_amp = active_device.type == "cuda" and dtype in {torch.float16, torch.bfloat16}

    every = max(1, int(args.log_interval_samples))
    for i, sample in enumerate(tqdm(samples, desc=f"{label} eval", ncols=100), start=1):
        if i == 1 or i % every == 0 or i == total:
            LOGGER.info("Progress %s: %d/%d", label, i, total)
        model_inputs, render_used = build_inputs(
            tokenizer=tokenizer,
            question=sample.question,
            device=active_device,
            render_mode=args.render_mode,
            prompt_template=args.prompt_template,
            system_prompt=args.system_prompt,
        )
        input_ids = model_inputs["input_ids"]
        input_len = int(input_ids.shape[1])

        if render_used == "chat":
            chat_render_count += 1
        else:
            plain_render_count += 1

        gen_kwargs: dict[str, Any] = {
            "max_new_tokens": int(args.max_new_tokens),
            "do_sample": do_sample,
            "repetition_penalty": float(args.repetition_penalty),
            "eos_token_id": eos_token_id,
            "pad_token_id": pad_token_id,
        }
        if do_sample:
            gen_kwargs["temperature"] = float(args.temperature)
            gen_kwargs["top_p"] = float(args.top_p)

        started = time.perf_counter()
        output_ids = None
        retry_tokens = int(args.max_new_tokens)
        retries_left = max(0, int(args.oom_retry_halves))
        while True:
            try:
                gen_kwargs["max_new_tokens"] = retry_tokens
                with torch.inference_mode():
                    if use_amp:
                        with torch.amp.autocast(device_type="cuda", dtype=dtype):
                            output_ids = model.generate(**model_inputs, **gen_kwargs)
                    else:
                        output_ids = model.generate(**model_inputs, **gen_kwargs)
                break
            except torch.OutOfMemoryError as oom:
                if active_device.type != "cuda":
                    raise
                LOGGER.warning(
                    "CUDA OOM at %s sample %d/%d with max_new_tokens=%d: %s",
                    label,
                    i,
                    total,
                    retry_tokens,
                    oom,
                )
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                if retries_left > 0 and retry_tokens > 16:
                    retry_tokens = max(16, retry_tokens // 2)
                    retries_left -= 1
                    LOGGER.warning("Retrying with reduced max_new_tokens=%d", retry_tokens)
                    continue
                if args.oom_fallback_to_cpu:
                    LOGGER.warning("Falling back to CPU for %s after CUDA OOM.", label)
                    model.to(torch.device("cpu"))
                    active_device = torch.device("cpu")
                    use_amp = False
                    model_inputs = {
                        k: v.to(active_device) if isinstance(v, torch.Tensor) else v
                        for k, v in model_inputs.items()
                    }
                    continue
                raise
        assert output_ids is not None
        elapsed_s = max(1e-9, time.perf_counter() - started)

        gen_ids = output_ids[:, input_len:]
        generated_tokens = int(gen_ids.shape[1])
        total_generated_tokens += generated_tokens
        total_latency_s += elapsed_s
        if generated_tokens >= int(args.max_new_tokens):
            truncation_hits += 1

        generated_text = tokenizer.decode(gen_ids[0].detach().cpu().tolist(), skip_special_tokens=True)
        pred_answer = extract_final_answer(generated_text)
        if not pred_answer.strip():
            empty_prediction_count += 1

        gold_answer = sample.answer.strip()
        norm_exact = normalize_text(pred_answer) == normalize_text(gold_answer) and bool(normalize_text(gold_answer))
        if norm_exact:
            normalized_exact_count += 1

        pred_num = parse_number(pred_answer)
        gold_num = parse_number(gold_answer)
        abs_error: float | None = None
        numeric_tol = False
        if gold_num is not None:
            numeric_total += 1
            if pred_num is not None:
                abs_error = abs(pred_num - gold_num)
                numeric_abs_error_sum += abs_error
                numeric_abs_error_count += 1
                numeric_tol = abs_error <= float(args.numeric_tolerance)
                if numeric_tol:
                    numeric_tol_count += 1

        records.append(
            {
                "sample_id": sample.sample_id,
                "question": sample.question,
                "gold_answer": gold_answer,
                "gold_raw_answer": sample.raw_answer,
                "prediction": pred_answer,
                "raw_generation": generated_text,
                "normalized_exact_match": norm_exact,
                "numeric_within_tolerance": numeric_tol,
                "abs_numeric_error": abs_error,
                "generated_tokens": generated_tokens,
                "latency_ms": elapsed_s * 1000.0,
                "render_mode_used": render_used,
            }
        )

    summary = {
        "model_label": label,
        "model_id": model_id,
        "num_samples": total,
        "normalized_exact_match_accuracy": (normalized_exact_count / total) if total else 0.0,
        "numeric_total": numeric_total,
        "numeric_within_tolerance_accuracy": (numeric_tol_count / numeric_total) if numeric_total else 0.0,
        "avg_abs_numeric_error": (numeric_abs_error_sum / numeric_abs_error_count) if numeric_abs_error_count else None,
        "avg_generated_tokens": (total_generated_tokens / total) if total else 0.0,
        "truncation_rate": (truncation_hits / total) if total else 0.0,
        "empty_prediction_rate": (empty_prediction_count / total) if total else 0.0,
        "avg_latency_ms": (total_latency_s * 1000.0 / total) if total else 0.0,
        "tokens_per_second": (total_generated_tokens / total_latency_s) if total_latency_s > 0 else 0.0,
        "chat_render_rate": (chat_render_count / total) if total else 0.0,
        "plain_render_rate": (plain_render_count / total) if total else 0.0,
    }

    del model
    del tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    LOGGER.info(
        "Completed eval for %s: normalized_exact_match_accuracy=%.4f numeric_within_tolerance_accuracy=%.4f",
        label,
        summary["normalized_exact_match_accuracy"],
        summary["numeric_within_tolerance_accuracy"],
    )
    return summary, records


def build_head_to_head(
    teacher_records: list[dict[str, Any]],
    student_records: list[dict[str, Any]],
) -> dict[str, Any]:
    teacher_wins = 0
    student_wins = 0
    both_correct = 0
    both_wrong = 0

    for t_row, s_row in zip(teacher_records, student_records):
        t_ok = bool(t_row.get("normalized_exact_match"))
        s_ok = bool(s_row.get("normalized_exact_match"))
        if t_ok and not s_ok:
            teacher_wins += 1
        elif s_ok and not t_ok:
            student_wins += 1
        elif t_ok and s_ok:
            both_correct += 1
        else:
            both_wrong += 1

    total = len(teacher_records)
    return {
        "total": total,
        "teacher_wins": teacher_wins,
        "student_wins": student_wins,
        "both_correct": both_correct,
        "both_wrong": both_wrong,
        "teacher_win_rate": (teacher_wins / total) if total else 0.0,
        "student_win_rate": (student_wins / total) if total else 0.0,
    }


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def fmt(value: float | None, as_pct: bool = False) -> str:
    if value is None:
        return "NA"
    if as_pct:
        return f"{value * 100:.2f}%"
    if abs(value) >= 1000:
        return f"{value:,.2f}"
    if abs(value) >= 10:
        return f"{value:.3f}"
    return f"{value:.6f}"


def build_report(
    *,
    dataset_source: str,
    split: str,
    max_samples: int,
    teacher_summary: dict[str, Any],
    student_summary: dict[str, Any],
    head_to_head: dict[str, Any],
    args: argparse.Namespace,
) -> str:
    lines: list[str] = []
    lines.append("# GSM8K Teacher vs Student (Before Distillation)")
    lines.append("")
    lines.append(f"- Generated at (UTC): `{utc_now()}`")
    lines.append(f"- Dataset source: `{dataset_source}`")
    lines.append(f"- Split: `{split}`")
    lines.append(f"- Evaluated samples: `{teacher_summary['num_samples']}`")
    lines.append(f"- Requested max samples: `{max_samples}`")
    lines.append(f"- Teacher model: `{teacher_summary['model_id']}`")
    lines.append(f"- Student model: `{student_summary['model_id']}`")
    lines.append("")

    lines.append("## Decoding")
    lines.append("")
    lines.append("| Key | Value |")
    lines.append("|---|---|")
    lines.append(f"| `max_new_tokens` | `{args.max_new_tokens}` |")
    lines.append(f"| `temperature` | `{args.temperature}` |")
    lines.append(f"| `top_p` | `{args.top_p}` |")
    lines.append(f"| `repetition_penalty` | `{args.repetition_penalty}` |")
    lines.append(f"| `render_mode` | `{args.render_mode}` |")
    lines.append(f"| `numeric_tolerance` | `{args.numeric_tolerance}` |")
    lines.append("")

    lines.append("## Side-by-Side Metrics")
    lines.append("")
    lines.append("| Metric | Teacher | Student |")
    lines.append("|---|---:|---:|")
    lines.append(
        "| `normalized_exact_match_accuracy` "
        f"| {fmt(teacher_summary['normalized_exact_match_accuracy'], as_pct=True)} "
        f"| {fmt(student_summary['normalized_exact_match_accuracy'], as_pct=True)} |"
    )
    lines.append(
        "| `numeric_within_tolerance_accuracy` "
        f"| {fmt(teacher_summary['numeric_within_tolerance_accuracy'], as_pct=True)} "
        f"| {fmt(student_summary['numeric_within_tolerance_accuracy'], as_pct=True)} |"
    )
    lines.append(
        "| `avg_abs_numeric_error` "
        f"| {fmt(teacher_summary['avg_abs_numeric_error'])} "
        f"| {fmt(student_summary['avg_abs_numeric_error'])} |"
    )
    lines.append(
        "| `avg_generated_tokens` "
        f"| {fmt(teacher_summary['avg_generated_tokens'])} "
        f"| {fmt(student_summary['avg_generated_tokens'])} |"
    )
    lines.append(
        "| `truncation_rate` "
        f"| {fmt(teacher_summary['truncation_rate'], as_pct=True)} "
        f"| {fmt(student_summary['truncation_rate'], as_pct=True)} |"
    )
    lines.append(
        "| `avg_latency_ms` "
        f"| {fmt(teacher_summary['avg_latency_ms'])} "
        f"| {fmt(student_summary['avg_latency_ms'])} |"
    )
    lines.append(
        "| `tokens_per_second` "
        f"| {fmt(teacher_summary['tokens_per_second'])} "
        f"| {fmt(student_summary['tokens_per_second'])} |"
    )
    lines.append("")

    lines.append("## Head-to-Head")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|---|---:|")
    lines.append(f"| Teacher wins | {head_to_head['teacher_wins']} |")
    lines.append(f"| Student wins | {head_to_head['student_wins']} |")
    lines.append(f"| Both correct | {head_to_head['both_correct']} |")
    lines.append(f"| Both wrong | {head_to_head['both_wrong']} |")
    lines.append(f"| Teacher win rate | {fmt(head_to_head['teacher_win_rate'], as_pct=True)} |")
    lines.append(f"| Student win rate | {fmt(head_to_head['student_win_rate'], as_pct=True)} |")
    lines.append("")

    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    configure_logging(args.debug)
    if "{question}" not in args.prompt_template:
        raise ValueError("--prompt-template must contain {question}")

    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    LOGGER.info("Starting run with args=%s", vars(args))

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    samples, dataset_source = load_gsm8k_samples(args.dataset_path, args.split, args.max_samples)

    device = get_device(args.device)
    dtype = get_dtype(args.dtype, device)
    LOGGER.info("Resolved device=%s dtype=%s", device, dtype)

    teacher_summary, teacher_records = evaluate_one_model(
        label="teacher",
        model_id=args.teacher_model,
        samples=samples,
        args=args,
        device=device,
        dtype=dtype,
    )
    student_summary, student_records = evaluate_one_model(
        label="student",
        model_id=args.student_model,
        samples=samples,
        args=args,
        device=device,
        dtype=dtype,
    )

    head_to_head = build_head_to_head(teacher_records, student_records)

    combined_rows: list[dict[str, Any]] = []
    for sample, teacher_row, student_row in zip(samples, teacher_records, student_records):
        combined_rows.append(
            {
                "sample_id": sample.sample_id,
                "question": sample.question,
                "gold_answer": sample.answer,
                "gold_raw_answer": sample.raw_answer,
                "teacher": teacher_row,
                "student": student_row,
            }
        )

    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.output_dir) if args.output_dir else Path("newoutput") / "evals" / f"beforeDistillation.{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)

    comparison_payload = {
        "created_at_utc": utc_now(),
        "dataset_source": dataset_source,
        "dataset_split": args.split,
        "num_samples": len(samples),
        "device": str(device),
        "dtype": str(dtype),
        "settings": {
            "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "repetition_penalty": args.repetition_penalty,
            "render_mode": args.render_mode,
            "numeric_tolerance": args.numeric_tolerance,
            "seed": args.seed,
        },
        "teacher_summary": teacher_summary,
        "student_summary": student_summary,
        "head_to_head": head_to_head,
    }

    (out_dir / "comparison.json").write_text(json.dumps(comparison_payload, indent=2), encoding="utf-8")
    write_jsonl(out_dir / "predictions.jsonl", combined_rows)
    write_jsonl(out_dir / "teacher_predictions.jsonl", teacher_records)
    write_jsonl(out_dir / "student_predictions.jsonl", student_records)
    (out_dir / "report.md").write_text(
        build_report(
            dataset_source=dataset_source,
            split=args.split,
            max_samples=args.max_samples,
            teacher_summary=teacher_summary,
            student_summary=student_summary,
            head_to_head=head_to_head,
            args=args,
        ),
        encoding="utf-8",
    )

    LOGGER.info("Artifacts written to %s", out_dir)
    print(f"[DONE] comparison={out_dir / 'comparison.json'}")
    print(f"[DONE] predictions={out_dir / 'predictions.jsonl'}")
    print(f"[DONE] report={out_dir / 'report.md'}")
    print(
        "[RESULT] teacher_normalized_exact_match_accuracy="
        f"{teacher_summary['normalized_exact_match_accuracy']:.4f}"
    )
    print(
        "[RESULT] student_normalized_exact_match_accuracy="
        f"{student_summary['normalized_exact_match_accuracy']:.4f}"
    )


if __name__ == "__main__":
    main()
