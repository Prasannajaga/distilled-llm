from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
DATASETS_ROOT = PROJECT_ROOT / "datasets"
if str(DATASETS_ROOT) not in sys.path:
    sys.path.insert(0, str(DATASETS_ROOT))

from Cdatasets.tokenizer import load_tokenizer
from scripts.infer import (
    _as_batched_input_ids,
    apply_inference_overrides,
    build_model,
    find_checkpoint,
    find_output_dir,
    load_saved_config,
    load_state_dict_with_tril_fallback,
    prepare_input_ids,
)
from utils.common import get_device, params


@dataclass
class EvalSample:
    sample_id: int
    question: str
    answer: str
    category: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a single model checkpoint on a math dataset.")
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to output model directory (example: ./output/mini-math-teacher-sft-8K).",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        dest="model_name",
        default=None,
        help="Optional explicit checkpoint file (.pt). Defaults to latest checkpoint in --output.",
    )
    parser.add_argument(
        "--eval-file",
        type=str,
        default="data/math_eval_100.jsonl",
        help="Evaluation dataset path (jsonl/json).",
    )
    parser.add_argument("--output-dir", type=str, default=None, help="Directory for eval artifacts.")
    parser.add_argument("--device", type=str, default="auto", help="Device: auto, cuda, mps, cpu.")
    parser.add_argument("--max-samples", type=int, default=None, help="Optional cap on number of samples.")
    parser.add_argument("--seed", type=int, default=42)

    # These names intentionally match scripts.infer.apply_inference_overrides
    parser.add_argument("--max_new_tokens", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_k", type=int, default=None)
    parser.add_argument("--stop_on_eos", type=int, choices=[0, 1], default=1)
    parser.add_argument("--repetition_penalty", type=float, default=1.0)

    parser.add_argument("--numeric-tolerance", type=float, default=1e-3)
    parser.add_argument(
        "--render-mode",
        type=str,
        choices=["auto", "chat", "plain"],
        default="auto",
        help="Prompt rendering mode. auto prefers chat template if available.",
    )
    parser.add_argument(
        "--prompt-template",
        type=str,
        default="Question: {question}\nAnswer:",
        help="Fallback prompt template for plain mode. Must contain {question}.",
    )
    parser.add_argument(
        "--system-prompt",
        type=str,
        default="You are a math assistant. Return only the final numeric answer.",
        help="System prompt used in chat render mode.",
    )
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_eval_row(row: dict[str, Any]) -> tuple[str, str, str] | None:
    q_keys = ("question", "problem", "prompt", "input", "instruction")
    a_keys = ("answer", "final_answer", "target", "output", "label")

    question = ""
    for key in q_keys:
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            question = value.strip()
            break

    answer = ""
    for key in a_keys:
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            answer = value.strip()
            break

    if not answer:
        solution = row.get("solution", row.get("chain_of_thought", ""))
        if isinstance(solution, str) and solution.strip():
            lines = [line.strip() for line in solution.splitlines() if line.strip()]
            if lines:
                answer = lines[-1]

    category = str(row.get("category", "uncategorized")).strip() or "uncategorized"
    if not question or not answer:
        return None
    return question, answer, category


def load_eval_samples(eval_file: str, max_samples: int | None) -> tuple[list[EvalSample], str]:
    path = Path(eval_file)
    if not path.exists():
        raise FileNotFoundError(f"Eval file not found: {eval_file}")

    rows: list[dict[str, Any]] = []
    if path.suffix.lower() == ".jsonl":
        for line in path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    else:
        payload = json.loads(path.read_text())
        if isinstance(payload, list):
            rows = payload
        elif isinstance(payload, dict):
            for key in ("samples", "data", "items", "examples"):
                value = payload.get(key)
                if isinstance(value, list):
                    rows = value
                    break
        if not rows:
            raise ValueError(
                "Could not parse eval JSON. Expected list or dict with samples/data/items/examples list."
            )

    samples: list[EvalSample] = []
    for idx, row in enumerate(rows):
        if not isinstance(row, dict):
            continue
        parsed = parse_eval_row(row)
        if not parsed:
            continue
        question, answer, category = parsed
        raw_id = row.get("id", idx + 1)
        try:
            sample_id = int(raw_id)
        except (TypeError, ValueError):
            sample_id = idx + 1
        samples.append(
            EvalSample(
                sample_id=sample_id,
                question=question,
                answer=answer,
                category=category,
            )
        )
        if max_samples is not None and len(samples) >= max_samples:
            break

    if not samples:
        raise ValueError("No valid eval rows found with question+answer fields.")
    return samples, str(path)


def normalize_text(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"\\boxed\{([^{}]+)\}", r"\1", text)
    text = text.replace(",", "")
    text = re.sub(r"[^a-z0-9\.\-\/ ]+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


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


def _prepare_chat_input_ids(
    *,
    tokenizer: Any,
    question: str,
    system_prompt: str,
    device: torch.device,
) -> torch.Tensor:
    messages: list[dict[str, str]] = []
    if system_prompt.strip():
        messages.append({"role": "system", "content": system_prompt.strip()})
    messages.append({"role": "user", "content": question})

    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    )
    return _as_batched_input_ids(inputs, device, tokenizer=tokenizer)


def prepare_eval_input_ids(
    *,
    tokenizer: Any,
    question: str,
    device: torch.device,
    render_mode: str,
    prompt_template: str,
    system_prompt: str,
) -> tuple[torch.Tensor, str]:
    can_chat = hasattr(tokenizer, "apply_chat_template")
    if render_mode in {"auto", "chat"} and can_chat:
        try:
            return (
                _prepare_chat_input_ids(
                    tokenizer=tokenizer,
                    question=question,
                    system_prompt=system_prompt,
                    device=device,
                ),
                "chat",
            )
        except Exception:
            if render_mode == "chat":
                raise

    prompt = prompt_template.format(question=question)
    return prepare_input_ids(tokenizer, prompt, device), "plain"


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_report(
    *,
    summary: dict[str, Any],
    category_metrics: dict[str, dict[str, Any]],
    settings: dict[str, Any],
    dataset_source: str,
) -> str:
    lines: list[str] = []
    lines.append("# Math Eval Report")
    lines.append("")
    lines.append(f"- Generated at (UTC): `{utc_now()}`")
    lines.append(f"- Eval source: `{dataset_source}`")
    lines.append(f"- Samples: `{summary['num_samples']}`")
    lines.append(f"- Accuracy (normalized exact): `{summary['normalized_exact_match_accuracy']:.4f}`")
    lines.append(f"- Numeric within tolerance: `{summary['numeric_within_tolerance_accuracy']:.4f}`")
    lines.append("")
    lines.append("## Decoding Settings")
    lines.append("")
    lines.append("| Key | Value |")
    lines.append("|---|---|")
    for key, value in settings.items():
        lines.append(f"| `{key}` | `{value}` |")
    lines.append("")
    lines.append("## Category Metrics")
    lines.append("")
    lines.append("| Category | Total | Normalized EM | Numeric Tol Acc |")
    lines.append("|---|---:|---:|---:|")
    for category in sorted(category_metrics):
        row = category_metrics[category]
        lines.append(
            f"| {category} | {row['total']} | {row['normalized_exact_match_accuracy']:.4f} | {row['numeric_within_tolerance_accuracy']:.4f} |"
        )
    lines.append("")
    return "\n".join(lines)


def evaluate(
    *,
    model: torch.nn.Module,
    tokenizer: Any,
    samples: list[EvalSample],
    config: Any,
    device: torch.device,
    args: argparse.Namespace,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]], list[dict[str, Any]]]:
    use_amp = bool(getattr(config, "use_amp", False) and device.type == "cuda")
    amp_dtype = (
        torch.bfloat16
        if str(getattr(config, "amp_dtype", "")).lower() in {"bf16", "bfloat16"}
        else torch.float16
    )
    eos_token_id = (
        tokenizer.eos_token_id
        if bool(args.stop_on_eos) and getattr(tokenizer, "eos_token_id", None) is not None
        else None
    )

    total = len(samples)
    normalized_exact_match_count = 0
    numeric_total = 0
    numeric_tol_count = 0
    numeric_abs_error_sum = 0.0
    numeric_abs_error_count = 0
    total_generated_tokens = 0
    total_latency_s = 0.0
    truncation_hits = 0
    plain_render_count = 0
    chat_render_count = 0
    empty_prediction_count = 0

    cat_total: dict[str, int] = defaultdict(int)
    cat_norm_exact: dict[str, int] = defaultdict(int)
    cat_numeric_total: dict[str, int] = defaultdict(int)
    cat_numeric_tol: dict[str, int] = defaultdict(int)

    records: list[dict[str, Any]] = []
    model.eval()

    for sample in tqdm(samples, desc="math_eval", ncols=100):
        input_ids, render_used = prepare_eval_input_ids(
            tokenizer=tokenizer,
            question=sample.question,
            device=device,
            render_mode=args.render_mode,
            prompt_template=args.prompt_template,
            system_prompt=args.system_prompt,
        )
        input_len = int(input_ids.shape[1])
        if render_used == "chat":
            chat_render_count += 1
        else:
            plain_render_count += 1

        started = time.perf_counter()
        with torch.inference_mode():
            with torch.amp.autocast(
                enabled=use_amp,
                device_type=device.type,
                dtype=amp_dtype,
            ):
                output_ids = model.generate(
                    input_ids,
                    max_new_tokens=int(args.max_new_tokens),
                    temperature=float(args.temperature),
                    top_k=args.top_k,
                    repetition_penalty=float(args.repetition_penalty),
                    eos_token_id=eos_token_id,
                )
        elapsed_s = max(1e-9, time.perf_counter() - started)

        gen_ids = output_ids[:, input_len:]
        generated_tokens = int(gen_ids.shape[1])
        total_generated_tokens += generated_tokens
        total_latency_s += elapsed_s
        if generated_tokens >= int(args.max_new_tokens):
            truncation_hits += 1

        gen_tokens = gen_ids[0].detach().cpu().tolist()
        try:
            generated_text = tokenizer.decode(gen_tokens, skip_special_tokens=True)
        except TypeError:
            generated_text = tokenizer.decode(gen_tokens)
        pred_answer = extract_final_answer(generated_text)
        if not pred_answer.strip():
            empty_prediction_count += 1

        gold_answer = sample.answer.strip()
        norm_exact = normalize_text(pred_answer) == normalize_text(gold_answer) and bool(normalize_text(gold_answer))
        if norm_exact:
            normalized_exact_match_count += 1

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

        cat = sample.category
        cat_total[cat] += 1
        if norm_exact:
            cat_norm_exact[cat] += 1
        if gold_num is not None:
            cat_numeric_total[cat] += 1
        if numeric_tol:
            cat_numeric_tol[cat] += 1

        records.append(
            {
                "sample_id": sample.sample_id,
                "category": sample.category,
                "question": sample.question,
                "gold_answer": gold_answer,
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

    category_metrics: dict[str, dict[str, Any]] = {}
    for category, count in cat_total.items():
        numeric_count = cat_numeric_total.get(category, 0)
        category_metrics[category] = {
            "total": count,
            "normalized_exact_match_accuracy": cat_norm_exact.get(category, 0) / count if count else 0.0,
            "numeric_total": numeric_count,
            "numeric_within_tolerance_accuracy": (
                cat_numeric_tol.get(category, 0) / numeric_count
            )
            if numeric_count
            else 0.0,
        }

    summary = {
        "num_samples": total,
        "normalized_exact_match_accuracy": (normalized_exact_match_count / total) if total else 0.0,
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
    return summary, category_metrics, records


def main() -> None:
    args = parse_args()
    if "{question}" not in args.prompt_template:
        raise ValueError("--prompt-template must contain {question}")

    torch.manual_seed(args.seed)
    random.seed(args.seed)

    output_dir = find_output_dir(args.output)
    checkpoint_path = find_checkpoint(output_dir, args.model_name)

    config = load_saved_config(output_dir)
    apply_inference_overrides(config, args)

    requested_device = args.device if args.device != "auto" else "auto"
    device = get_device(requested_device)
    if device.type != "cuda":
        config.use_amp = False

    tokenizer = load_tokenizer(str(output_dir))
    model = build_model(config, vocab_size=len(tokenizer)).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    load_state_dict_with_tril_fallback(model, state_dict)
    model.eval()

    samples, dataset_source = load_eval_samples(args.eval_file, args.max_samples)
    summary, category_metrics, records = evaluate(
        model=model,
        tokenizer=tokenizer,
        samples=samples,
        config=config,
        device=device,
        args=args,
    )

    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    default_out = Path("output") / "evals" / f"math_eval_{ts}"
    out_dir = Path(args.output_dir) if args.output_dir else default_out
    out_dir.mkdir(parents=True, exist_ok=True)

    settings = {
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_k": args.top_k,
        "repetition_penalty": args.repetition_penalty,
        "stop_on_eos": args.stop_on_eos,
        "render_mode": args.render_mode,
        "numeric_tolerance": args.numeric_tolerance,
    }

    payload = {
        "created_at_utc": utc_now(),
        "dataset_source": dataset_source,
        "model": {
            "output_dir": str(output_dir),
            "checkpoint_path": str(checkpoint_path),
            "parameters": params(model),
        },
        "settings": settings,
        "summary": summary,
        "category_metrics": category_metrics,
    }
    (out_dir / "summary.json").write_text(json.dumps(payload, indent=2))
    write_jsonl(out_dir / "predictions.jsonl", records)
    (out_dir / "report.md").write_text(
        build_report(
            summary=summary,
            category_metrics=category_metrics,
            settings=settings,
            dataset_source=dataset_source,
        )
    )

    print(f"[MODEL] output={output_dir}")
    print(f"[CKPT] {checkpoint_path}")
    print(f"[DONE] summary={out_dir / 'summary.json'}")
    print(f"[DONE] predictions={out_dir / 'predictions.jsonl'}")
    print(f"[DONE] report={out_dir / 'report.md'}")
    print(f"[RESULT] normalized_exact_match_accuracy={summary['normalized_exact_match_accuracy']:.4f}")
    print(f"[RESULT] numeric_within_tolerance_accuracy={summary['numeric_within_tolerance_accuracy']:.4f}")


if __name__ == "__main__":
    main()
