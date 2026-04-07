#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import os
import re
import shutil
import subprocess
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from datasets import Dataset, load_dataset
from newScripts.common import (
    ANSWER_CONTRACT_INSTRUCTION,
    ANSWER_FIELD,
    BUCKET_BOTH_CORRECT,
    BUCKET_BOTH_WRONG,
    BUCKET_TEACHER_CORRECT_STUDENT_WRONG,
    BUCKET_TEACHER_WRONG_STUDENT_CORRECT,
    EXAMPLES_KEY,
    GOLD_ANSWER_FIELD,
    GOLD_FIELD,
    QUESTION_FIELD,
    SAMPLE_COMPARISON_KEY,
    STUDENT_EXTRACTED_FIELD,
    STUDENT_MATCH_FIELD,
    STUDENT_PRED_FIELD,
    STUDENT_RAW_FIELD,
    STUDENT_RAW_PREDICTION_FIELD,
    STUDENT_OK_FIELD,
    TEACHER_EXTRACTED_FIELD,
    TEACHER_MATCH_FIELD,
    TEACHER_PRED_FIELD,
    TEACHER_RAW_FIELD,
    TEACHER_RAW_PREDICTION_FIELD,
    TEACHER_OK_FIELD,
    WINNER_FIELD,
    WINNER_LABEL_FIELD,
    answers_match,
    compute_bucket_metrics,
    compute_overlap_report,
    configure_logging as common_configure_logging,
    extract_priority_answer,
    log_state as common_log_state,
    read_json,
    write_jsonl_rows,
)

DEFAULT_STUDENT_MODEL = "/media/prasanna/716F26140AED9B67/models/models--Qwen--Qwen2-0.5B-Instruct"
DEFAULT_TEACHER_MODEL = "/media/prasanna/716F26140AED9B67/models/models--Qwen--Qwen2-Math-1.5B-Instruct"
DEFAULT_GSM8K_PATH = "/media/prasanna/716F26140AED9B67/datasets/GSM8K"
LOGGER = logging.getLogger("run_lm_eval_custom")
COMMAND_TAIL_LINES = 25
HEARTBEAT_SECONDS = 30


def configure_logging() -> None:
    global LOGGER
    LOGGER = common_configure_logging("run_lm_eval_custom")


def log_state(state: str, **fields: Any) -> None:
    common_log_state(LOGGER, state, **fields)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run custom lm-eval benchmark on local GSM8K-style dataset")
    p.add_argument("--dataset-path", type=str, default=DEFAULT_GSM8K_PATH)
    p.add_argument(
        "--dataset-jsonl",
        type=str,
        default=None,
        help="Optional direct JSONL dataset path with question/answer fields. If set, bypasses dataset-path/split loading.",
    )
    p.add_argument("--split", choices=["train", "test"], default="test")
    p.add_argument("--max-samples", type=int, default=0)
    p.add_argument("--student-model", type=str, default=DEFAULT_STUDENT_MODEL)
    p.add_argument(
        "--reuse-student-comparison-json",
        type=str,
        default=None,
        help="Optional path to an existing comparison.json to reuse student artifacts/scores without re-running student lm-eval.",
    )
    p.add_argument(
        "--resume-student-only",
        action="store_true",
        help="Skip teacher lm-eval and run only student lm-eval in an existing eval folder.",
    )
    p.add_argument(
        "--reuse-teacher-comparison-json",
        type=str,
        default=None,
        help="Optional path to an existing comparison.json to reuse teacher artifacts/scores without re-running teacher lm-eval.",
    )
    p.add_argument(
        "--reuse-dataset-jsonl-from-comparison",
        type=str,
        default=None,
        help="Optional comparison.json whose dataset_jsonl should be reused exactly for this run.",
    )
    p.add_argument("--teacher-model", type=str, default=DEFAULT_TEACHER_MODEL)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--batch-size", type=str, default="1")
    p.add_argument("--limit", type=str, default=None, help="lm-eval limit, e.g. 100 or 0.1")
    p.add_argument(
        "--allow-env-limit",
        action="store_true",
        help="If set and --limit is not passed, allow LIMIT/EVAL_LIMIT env var to provide lm-eval limit.",
    )
    p.add_argument(
        "--allow-partial-eval",
        action="store_true",
        help="If set, do not hard-fail when evaluated rows are smaller than expected rows.",
    )
    p.add_argument(
        "--expected-rows",
        type=int,
        default=0,
        help="Optional explicit expected row count for eval-integrity checks (<=0 uses dataset rows).",
    )
    p.add_argument(
        "--overlap-reference",
        action="append",
        default=[],
        help="Reference split for overlap check, format name=path (path may be jsonl or comparison.json).",
    )
    p.add_argument(
        "--overlap-threshold",
        type=float,
        default=0.0,
        help="Max allowed overlap fraction for eval-vs-reference checks (default 0.0).",
    )
    p.add_argument(
        "--allow-overlap",
        action="store_true",
        help="Allow overlap above threshold (still logs loud warning and writes overlap report).",
    )
    p.add_argument("--num-fewshot", type=int, default=8)
    p.add_argument("--gen-max-toks", type=int, default=256)
    p.add_argument("--retry-count", type=int, default=1, help="Retries per lm-eval subprocess command.")
    p.add_argument(
        "--teacher-fallback-device",
        type=str,
        default="cuda:0",
        help="Fallback device if teacher lm-eval command fails after retries.",
    )
    p.add_argument(
        "--teacher-fallback-max-gen-toks",
        type=int,
        default=128,
        help="Fallback max_gen_toks if teacher lm-eval command fails after retries.",
    )
    p.add_argument(
        "--student-fallback-device",
        type=str,
        default="cuda:0",
        help="Fallback device if student lm-eval command fails after retries.",
    )
    p.add_argument(
        "--gpu-optimize-6gb",
        type=int,
        choices=[0, 1],
        default=1,
        help="If 1, apply RTX 4050 6GB-friendly CUDA retry tuning before any CPU fallback.",
    )
    p.add_argument(
        "--student-fallback-max-gen-toks",
        type=int,
        default=128,
        help="Fallback max_gen_toks if student lm-eval command fails after retries.",
    )
    p.add_argument(
        "--fewshot-cot",
        type=int,
        choices=[0, 1],
        default=1,
        help="If 1, few-shot exemplars include full GSM8K worked answers; if 0, numeric target only.",
    )
    p.add_argument(
        "--sample-comparison-limit",
        type=int,
        default=0,
        help="How many sample comparison rows to store in comparison.json (<=0 means all).",
    )
    p.add_argument(
        "--local-models-only",
        action="store_true",
        help="Force local model loading only (no Hugging Face download).",
    )
    p.add_argument(
        "--eval-name",
        type=str,
        default=None,
        help="Run name for output folder and task naming (no timestamp suffix auto-added).",
    )
    p.add_argument(
        "--stage",
        type=str,
        default="adhoc",
        help="Pipeline stage label, e.g. before-distill, teacher-sft, after-distill.",
    )
    p.add_argument(
        "--experiment",
        type=str,
        default="default",
        help="Experiment group name used for tracking and dashboards.",
    )
    p.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Optional stable run id for cross-stage linking.",
    )
    p.add_argument(
        "--parent-eval",
        type=str,
        default=None,
        help="Optional parent eval name (for lineage tracking).",
    )
    p.add_argument(
        "--notes",
        type=str,
        default="",
        help="Optional free-text notes attached to this eval run.",
    )
    p.add_argument(
        "--overwrite-output",
        action="store_true",
        help="Clear existing output dir before running (keeps stable naming; no auto timestamp suffix).",
    )
    p.add_argument("--output-dir", type=str, default=None)
    p.add_argument(
        "--echo-subprocess",
        type=int,
        choices=[0, 1],
        default=1,
        help="If 1, stream child lm-eval output live instead of only keeping it in the failure tail.",
    )
    p.add_argument("--render-html", type=int, choices=[0, 1], default=1, help="Render report.html at end of run")
    p.add_argument(
        "--html-limit",
        type=int,
        default=0,
        help="Max sample rows shown in report.html (<=0 means all).",
    )
    p.add_argument(
        "--html-per-eval-sample-limit",
        type=int,
        default=0,
        help="Max loaded rows per eval for all-evals sample explorer in report.html (<=0 means all).",
    )
    p.add_argument("--html-title", type=str, default="lm-eval Dashboard", help="HTML report title")
    p.add_argument(
        "--fast-mode",
        action="store_true",
        help="Speed-oriented mode: disables HTML rendering. Does not change generation/scoring settings.",
    )
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def slugify(name: str) -> str:
    s = name.strip().lower()
    s = re.sub(r"[^a-z0-9._-]+", "-", s)
    s = re.sub(r"-{2,}", "-", s).strip("-")
    return s or "eval"


def ensure_unique_dir(base_root: Path, desired_name: str) -> Path:
    candidate = base_root / desired_name
    if not candidate.exists():
        return candidate
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    candidate = base_root / f"{desired_name}__{ts}"
    if not candidate.exists():
        return candidate
    idx = 2
    while True:
        c = base_root / f"{desired_name}__{ts}_{idx}"
        if not c.exists():
            return c
        idx += 1


def taskify(name: str) -> str:
    s = name.strip().lower()
    s = re.sub(r"[^a-z0-9_]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s or "eval"


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


def to_jsonl_from_arrow(split_arrow: Path, out_jsonl: Path, max_samples: int) -> int:
    ds = Dataset.from_file(str(split_arrow))
    count = 0
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with out_jsonl.open("w", encoding="utf-8") as f:
        for row in ds:
            q = str(row.get(QUESTION_FIELD, "")).strip()
            a = str(row.get(ANSWER_FIELD, "")).strip()
            if not q or not a:
                continue
            f.write(json.dumps({QUESTION_FIELD: q, ANSWER_FIELD: a}, ensure_ascii=False) + "\n")
            count += 1
            if max_samples > 0 and count >= max_samples:
                break
    return count


def to_jsonl_from_hf(split: str, out_jsonl: Path, max_samples: int) -> int:
    ds = load_dataset("openai/gsm8k", "main", split=split)
    count = 0
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with out_jsonl.open("w", encoding="utf-8") as f:
        for row in ds:
            q = str(row.get(QUESTION_FIELD, "")).strip()
            a = str(row.get(ANSWER_FIELD, "")).strip()
            if not q or not a:
                continue
            f.write(json.dumps({QUESTION_FIELD: q, ANSWER_FIELD: a}, ensure_ascii=False) + "\n")
            count += 1
            if max_samples > 0 and count >= max_samples:
                break
    return count


def copy_jsonl_dataset(src_jsonl: Path, out_jsonl: Path, max_samples: int) -> int:
    count = 0
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with src_jsonl.open("r", encoding="utf-8") as src, out_jsonl.open("w", encoding="utf-8") as dst:
        for line in src:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            q = str(row.get(QUESTION_FIELD, "")).strip()
            a = str(row.get(ANSWER_FIELD, "")).strip()
            if not q or not a:
                continue
            dst.write(json.dumps({QUESTION_FIELD: q, ANSWER_FIELD: a}, ensure_ascii=False) + "\n")
            count += 1
            if max_samples > 0 and count >= max_samples:
                break
    return count


def resolve_limit_config(
    cli_limit: str | None,
    *,
    allow_env_limit: bool,
    env: dict[str, str] | None = None,
) -> dict[str, Any]:
    raw_cli = str(cli_limit).strip() if cli_limit is not None else ""
    if raw_cli:
        return {
            "active": True,
            "value": raw_cli,
            "source": "cli",
            "ignored_env_limit": None,
        }
    env_map = env if env is not None else os.environ
    inherited = str(env_map.get("EVAL_LIMIT", "") or env_map.get("LIMIT", "")).strip()
    if inherited and allow_env_limit:
        return {
            "active": True,
            "value": inherited,
            "source": "env",
            "ignored_env_limit": None,
        }
    return {
        "active": False,
        "value": None,
        "source": "none",
        "ignored_env_limit": inherited or None,
    }


def parse_named_path(raw: str) -> tuple[str, Path]:
    text = str(raw or "").strip()
    if not text or "=" not in text:
        raise ValueError(f"Invalid --overlap-reference '{raw}'. Expected name=path.")
    name, path = text.split("=", 1)
    split_name = name.strip()
    path_text = path.strip()
    split_path = Path(path_text)
    if not split_name:
        raise ValueError(f"Invalid --overlap-reference '{raw}'. Empty name.")
    if not path_text:
        raise ValueError(f"Invalid --overlap-reference '{raw}'. Empty path.")
    return split_name, split_path


def _rows_from_comparison_payload(payload: dict[str, Any]) -> list[dict[str, Any]]:
    sample_cmp = payload.get(SAMPLE_COMPARISON_KEY, {})
    if isinstance(sample_cmp, dict):
        examples = sample_cmp.get(EXAMPLES_KEY, [])
        if isinstance(examples, list):
            return [dict(row) for row in examples if isinstance(row, dict)]
    return []


def load_rows_for_overlap(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Overlap reference file not found: {path}")
    if path.suffix.lower() == ".jsonl":
        rows = read_sample_rows(path, limit=None)
        return [dict(row) for row in rows]
    if path.suffix.lower() == ".json":
        payload = read_json(path)
        if not isinstance(payload, dict):
            raise RuntimeError(f"Invalid overlap JSON object: {path}")
        rows = _rows_from_comparison_payload(payload)
        if rows:
            return rows
        dataset_jsonl = payload.get("dataset_jsonl")
        if isinstance(dataset_jsonl, str) and dataset_jsonl.strip():
            return load_rows_for_overlap(Path(dataset_jsonl))
        if isinstance(payload.get("rows"), list):
            return [dict(row) for row in payload["rows"] if isinstance(row, dict)]
    raise RuntimeError(f"Unsupported overlap reference format (expected jsonl/comparison.json): {path}")


def evaluate_overlap_guardrails(
    *,
    eval_rows: list[dict[str, Any]],
    references: list[tuple[str, Path]],
    out_path: Path,
    threshold: float,
    allow_overlap: bool,
) -> dict[str, Any]:
    named_rows: dict[str, list[dict[str, Any]]] = {"eval": eval_rows}
    reference_files: dict[str, str] = {}
    for name, path in references:
        rows = load_rows_for_overlap(path)
        named_rows[name] = rows
        reference_files[name] = str(path)

    report = compute_overlap_report(named_rows)
    report["threshold"] = float(threshold)
    report["allow_overlap"] = bool(allow_overlap)
    report["reference_files"] = reference_files
    violations: list[dict[str, Any]] = []
    for pair in report.get("pairwise", []):
        if str(pair.get("left")) == "eval":
            overlap_pct = pair.get("overlap_pct_of_left")
        elif str(pair.get("right")) == "eval":
            overlap_pct = pair.get("overlap_pct_of_right")
        else:
            overlap_pct = None
        if isinstance(overlap_pct, (int, float)) and float(overlap_pct) > float(threshold):
            violations.append(
                {
                    "left": pair.get("left"),
                    "right": pair.get("right"),
                    "overlap_count": pair.get("overlap_count"),
                    "overlap_pct_eval": float(overlap_pct),
                }
            )
    report["violations"] = violations

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    if violations and not allow_overlap:
        first = violations[0]
        raise RuntimeError(
            "Overlap guardrail violation: eval split overlaps reference split above threshold. "
            f"threshold={threshold} left={first.get('left')} right={first.get('right')} "
            f"overlap_count={first.get('overlap_count')} overlap_pct_eval={first.get('overlap_pct_eval'):.6f}. "
            "Use --allow-overlap to override explicitly."
        )
    if violations and allow_overlap:
        LOGGER.warning(
            "Overlap guardrail override active. Violations=%s threshold=%.6f",
            len(violations),
            float(threshold),
        )
    return report


def infer_partial_eval_source(
    *,
    limit_cfg: dict[str, Any],
    sample_comparison_limit: int,
    reused_teacher: bool,
    reused_student: bool,
) -> str:
    if bool(limit_cfg.get("active")):
        return f"lm-eval --limit={limit_cfg.get('value')}"
    if reused_teacher or reused_student:
        return "reused comparison artifacts with fewer sample rows"
    if int(sample_comparison_limit) > 0:
        return f"sample comparison limit={int(sample_comparison_limit)}"
    return "lm-eval sample/result mismatch; check output_path artifacts"


def build_eval_integrity_status(
    *,
    expected_rows: int,
    loaded_rows: int,
    allow_partial_eval: bool,
    likely_source: str,
    limit_cfg: dict[str, Any],
) -> dict[str, Any]:
    return {
        "expected_rows": int(expected_rows),
        "loaded_rows": int(loaded_rows),
        "partial_eval_detected": int(loaded_rows) < int(expected_rows),
        "allow_partial_eval": bool(allow_partial_eval),
        "likely_mismatch_source": str(likely_source),
        "limit_status": {
            "active": bool(limit_cfg.get("active")),
            "value": limit_cfg.get("value"),
            "source": limit_cfg.get("source"),
        },
    }


def build_partial_eval_error_message(integrity: dict[str, Any]) -> str:
    return (
        "Invalid partial evaluation detected: loaded rows are smaller than expected rows. "
        f"expected_rows={integrity.get('expected_rows')} loaded_rows={integrity.get('loaded_rows')} "
        f"likely_source='{integrity.get('likely_mismatch_source')}'. "
        "This commonly happens when --limit is active or artifacts are reused from subset runs. "
        "Use --allow-partial-eval only for explicit diagnostics."
    )


def write_task_yaml(
    task_path: Path,
    data_jsonl: Path,
    num_fewshot: int,
    task_name: str,
    fewshot_cot: bool,
) -> str:
    doc_to_target = f"{{{{{ANSWER_FIELD}}}}}" if fewshot_cot else f"{{{{{ANSWER_FIELD}.split('####')[-1].strip()}}}}"
    yaml = f"""task: {task_name}
dataset_path: json
dataset_kwargs:
  data_files:
    test: {data_jsonl.as_posix()}
output_type: generate_until
test_split: test
training_split: null
validation_split: null
doc_to_text: |
  Instruction: {ANSWER_CONTRACT_INSTRUCTION}
  Q: {{{{{QUESTION_FIELD}}}}}

  A:
doc_to_target: "{doc_to_target}"
num_fewshot: {num_fewshot}
filter_list:
  - name: marker-priority
    filter:
      - function: regex
        group_select: -1
        regex_pattern: '(?is)The answer is\\s*(-?[0-9.,]+)|####\\s*(-?[0-9.,]+)|(-?[$0-9.,]{{2,}})|(-?[0-9]+)'
      - function: take_first
  - name: strict-match
    filter:
      - function: regex
        regex_pattern: 'The answer is (-?[0-9.,]+).'
      - function: take_first
  - name: flexible-extract
    filter:
      - function: regex
        group_select: -1
        regex_pattern: '(-?[$0-9.,]{{2,}})|(-?[0-9]+)'
      - function: take_first
generation_kwargs:
  do_sample: false
  until:
    - "Q:"
    - "</s>"
    - "<|im_end|>"
metric_list:
  - metric: exact_match
    aggregation: mean
    higher_is_better: true
    ignore_case: true
    ignore_punctuation: false
    regexes_to_ignore:
      - ","
      - '\\$'
      - '(?s).*#### '
      - '\\.$'
metadata:
  version: 1.0
"""
    task_path.write_text(yaml, encoding="utf-8")
    return task_name


def run_cmd(
    cmd: list[str],
    cwd: Path,
    retries: int = 0,
    env: dict[str, str] | None = None,
    *,
    echo_subprocess: bool,
) -> None:
    LOGGER.info("CMD: %s", " ".join(cmd))
    last_exc: subprocess.CalledProcessError | None = None
    for attempt in range(retries + 1):
        start = time.perf_counter()
        last_heartbeat = start
        tail: list[str] = []
        output_total = 0
        last_output_at = start
        try:
            proc = subprocess.Popen(
                cmd,
                cwd=str(cwd),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                env=env,
            )
            assert proc.stdout is not None
            def _reader() -> None:
                nonlocal output_total, last_output_at
                assert proc.stdout is not None
                for line in proc.stdout:
                    msg = line.rstrip("\n")
                    tail.append(msg)
                    output_total += 1
                    last_output_at = time.perf_counter()
                    if len(tail) > COMMAND_TAIL_LINES:
                        del tail[:-COMMAND_TAIL_LINES]
                    if echo_subprocess:
                        print(msg)

            t = threading.Thread(target=_reader, daemon=True)
            t.start()
            while proc.poll() is None:
                time.sleep(1.0)
                now = time.perf_counter()
                elapsed = now - start
                idle = now - last_output_at
                if idle >= HEARTBEAT_SECONDS and (now - last_heartbeat) >= HEARTBEAT_SECONDS:
                    LOGGER.info(
                        "Still running (%.1fs): output_lines_total=%d tail_lines=%d last_output=%.1fs_ago",
                        elapsed,
                        output_total,
                        len(tail),
                        idle,
                    )
                    last_heartbeat = now
            t.join(timeout=5)
            rc = proc.wait()
            if rc != 0:
                raise subprocess.CalledProcessError(rc, cmd)
            elapsed = time.perf_counter() - start
            LOGGER.info("Command succeeded in %.2fs", elapsed)
            return
        except subprocess.CalledProcessError as exc:
            last_exc = exc
            LOGGER.warning("Command failed (attempt %d/%d) rc=%s", attempt + 1, retries + 1, exc.returncode)
            if tail:
                LOGGER.warning("Last %d output lines:\n%s", len(tail), "\n".join(tail))
            if attempt < retries:
                continue
    assert last_exc is not None
    raise last_exc


def _replace_flag_value(cmd: list[str], flag: str, new_value: str) -> list[str]:
    out = cmd[:]
    if flag in out:
        i = out.index(flag)
        if i + 1 < len(out):
            out[i + 1] = new_value
    return out


def _append_model_args(cmd: list[str], extras: list[str]) -> list[str]:
    out = cmd[:]
    if "--model_args" not in out:
        return out
    i = out.index("--model_args")
    if i + 1 >= len(out):
        return out
    current = out[i + 1]
    parts = [p.strip() for p in current.split(",") if p.strip()]
    kv: dict[str, str] = {}
    ordered_keys: list[str] = []
    for p in parts:
        if "=" not in p:
            continue
        k, v = p.split("=", 1)
        k = k.strip()
        if not k:
            continue
        if k not in kv:
            ordered_keys.append(k)
        kv[k] = v.strip()
    for item in extras:
        if "=" not in item:
            continue
        key, val = item.split("=", 1)
        key = key.strip()
        val = val.strip()
        if not key:
            continue
        if key not in kv:
            ordered_keys.append(key)
        kv[key] = val
    out[i + 1] = ",".join([f"{k}={kv[k]}" for k in ordered_keys])
    return out


def _merge_model_arg_list(base: list[str], extras: list[str]) -> list[str]:
    merged = _append_model_args(["--model_args", ",".join(base)], extras)[1]
    return [p for p in merged.split(",") if p]


def ensure_lm_eval_installed() -> None:
    probe = subprocess.run(
        ["uv", "run", "python", "-c", "import lm_eval"],
        capture_output=True,
        text=True,
    )
    if probe.returncode != 0:
        raise RuntimeError(
            "Missing dependency: lm_eval is not installed in this uv environment.\n"
            "Run one of:\n"
            "  uv sync\n"
            "  uv pip install lm-eval\n"
        )


def read_sample_rows(samples_jsonl: Path, limit: int | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with samples_jsonl.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
            if limit is not None and limit > 0 and len(rows) >= limit:
                break
    return rows


def _latest_matching(output_root: Path, pattern: str) -> Path | None:
    candidates = sorted(
        output_root.rglob(pattern),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


def _remove_empty_parents(path: Path, stop_at: Path) -> None:
    cur = path.parent
    while cur != stop_at and stop_at in cur.parents:
        try:
            cur.rmdir()
        except OSError:
            break
        cur = cur.parent


def normalize_model_artifacts(model_out_dir: Path) -> tuple[Path | None, Path | None]:
    """
    Ensure only stable artifact names are used in model_out_dir:
      - results.json
      - samples.jsonl
    If timestamped/nested artifacts exist, move newest ones into stable paths.
    """
    stable_results = model_out_dir / "results.json"
    stable_samples = model_out_dir / "samples.jsonl"

    latest_results = _latest_matching(model_out_dir, "results*.json")
    latest_samples = _latest_matching(model_out_dir, "samples*.jsonl")

    if latest_results and latest_results.exists() and latest_results != stable_results:
        stable_results.parent.mkdir(parents=True, exist_ok=True)
        latest_results.replace(stable_results)
        _remove_empty_parents(latest_results, model_out_dir)
    if latest_samples and latest_samples.exists() and latest_samples != stable_samples:
        stable_samples.parent.mkdir(parents=True, exist_ok=True)
        latest_samples.replace(stable_samples)
        _remove_empty_parents(latest_samples, model_out_dir)

    # Cleanup any leftover timestamped duplicates.
    for p in list(model_out_dir.rglob("results*.json")):
        if p != stable_results and p.exists():
            p.unlink()
            _remove_empty_parents(p, model_out_dir)
    for p in list(model_out_dir.rglob("samples*.jsonl")):
        if p != stable_samples and p.exists():
            p.unlink()
            _remove_empty_parents(p, model_out_dir)

    return (
        stable_results if stable_results.exists() else None,
        stable_samples if stable_samples.exists() else None,
    )


def _filter_rank(filter_name: Any) -> int:
    name = str(filter_name or "").strip().lower()
    if name == "marker-priority":
        return 0
    if name == "flexible-extract":
        return 1
    if name == "strict-match":
        return 2
    return 10


def _sample_group_key(row: dict[str, Any]) -> str:
    for k in ("doc_hash", "prompt_hash", "target_hash"):
        v = row.get(k)
        if isinstance(v, str) and v:
            return f"{k}:{v}"
    return f"doc_id:{row.get('doc_id')}"


def _is_invalid_filtered(row: dict[str, Any]) -> bool:
    fr = row.get("filtered_resps")
    if isinstance(fr, list) and fr:
        return str(fr[0]).strip().lower() == "[invalid]"
    return False


def read_best_sample_rows(path: Path, limit_docs: int) -> list[dict[str, Any]]:
    best_by_doc: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            key = _sample_group_key(row)
            prev = best_by_doc.get(key)
            if prev is None:
                best_by_doc[key] = row
            else:
                prev_rank = (_filter_rank(prev.get("filter")), 1 if _is_invalid_filtered(prev) else 0)
                row_rank = (_filter_rank(row.get("filter")), 1 if _is_invalid_filtered(row) else 0)
                if row_rank < prev_rank:
                    best_by_doc[key] = row

    rows = list(best_by_doc.values())
    rows.sort(key=lambda r: int(r.get("doc_id", 10**9)) if str(r.get("doc_id", "")).isdigit() else 10**9)
    if limit_docs > 0:
        rows = rows[:limit_docs]
    return rows


def _extract_triplet(sample_row: dict[str, Any]) -> tuple[str, str, str, str]:
    doc = sample_row.get("doc", {}) if isinstance(sample_row.get("doc", {}), dict) else {}
    question = str(doc.get(QUESTION_FIELD, sample_row.get("prompt", "")))
    gold = str(sample_row.get("target", doc.get(ANSWER_FIELD, "")))

    raw_pred = ""
    resps = sample_row.get("resps")
    if isinstance(resps, list) and resps:
        first = resps[0]
        if isinstance(first, list) and first:
            raw_pred = str(first[0])
        else:
            raw_pred = str(first)

    extracted_pred = _extract_priority_answer(raw_pred)
    if not extracted_pred:
        fresps = sample_row.get("filtered_resps")
        if isinstance(fresps, list) and fresps:
            extracted_pred = str(fresps[0])
        if extracted_pred.strip().lower() == "[invalid]":
            extracted_pred = ""
    return question, gold, extracted_pred, raw_pred


def _extract_priority_answer(text: str) -> str:
    return extract_priority_answer(text)


def _matches(pred: str, gold: str) -> bool:
    return answers_match(pred, gold)


def build_sample_comparison(
    teacher_samples: Path | None,
    student_samples: Path | None,
    limit: int = 80,
) -> dict[str, Any]:
    if teacher_samples is None or student_samples is None:
        return {
            "teacher_samples_file": str(teacher_samples) if teacher_samples else None,
            "student_samples_file": str(student_samples) if student_samples else None,
            "loaded_rows": 0,
            "teacher_wins": 0,
            "student_wins": 0,
            "both_correct": 0,
            "both_wrong": 0,
            "teacher_accuracy": None,
            "student_accuracy": None,
            EXAMPLES_KEY: [],
        }

    t_rows = read_best_sample_rows(teacher_samples, limit_docs=limit)
    s_rows = read_best_sample_rows(student_samples, limit_docs=limit)
    n = min(len(t_rows), len(s_rows))

    teacher_wins = 0
    student_wins = 0
    both_correct = 0
    both_wrong = 0
    examples: list[dict[str, Any]] = []

    for i in range(n):
        tq, tg, tp_extracted, tp_raw = _extract_triplet(t_rows[i])
        sq, sg, sp_extracted, sp_raw = _extract_triplet(s_rows[i])
        question = tq or sq
        gold = tg or sg

        # Use extracted predictions for comparison; fall back to raw only if extraction failed.
        tp_for_match = tp_extracted if tp_extracted else tp_raw
        sp_for_match = sp_extracted if sp_extracted else sp_raw

        t_ok = _matches(tp_for_match, gold)
        s_ok = _matches(sp_for_match, gold)

        winner = "none"
        if t_ok and not s_ok:
            teacher_wins += 1
            winner = "teacher"
        elif s_ok and not t_ok:
            student_wins += 1
            winner = "student"
        elif t_ok and s_ok:
            both_correct += 1
            winner = "both"
        else:
            both_wrong += 1

        examples.append(
            {
                "idx": i,
                QUESTION_FIELD: question,
                GOLD_FIELD: gold,
                TEACHER_PRED_FIELD: tp_extracted,
                STUDENT_PRED_FIELD: sp_extracted,
                TEACHER_RAW_FIELD: tp_raw,
                STUDENT_RAW_FIELD: sp_raw,
                TEACHER_OK_FIELD: t_ok,
                STUDENT_OK_FIELD: s_ok,
                WINNER_FIELD: winner,
            }
        )

    return {
        "teacher_samples_file": str(teacher_samples),
        "student_samples_file": str(student_samples),
        "loaded_rows": n,
        "teacher_wins": teacher_wins,
        "student_wins": student_wins,
        "both_correct": both_correct,
        "both_wrong": both_wrong,
        "teacher_accuracy": ((teacher_wins + both_correct) / n) if n > 0 else None,
        "student_accuracy": ((student_wins + both_correct) / n) if n > 0 else None,
        EXAMPLES_KEY: examples,
    }


def write_sample_columns_jsonl(path: Path, sample_cmp: dict[str, Any]) -> None:
    """
    Write a flattened, analysis-friendly samples table with stable columns.
    """
    rows = sample_cmp.get(EXAMPLES_KEY, [])
    if not isinstance(rows, list):
        rows = []
    flat_rows = []
    for r in rows:
        flat_rows.append(
            {
                QUESTION_FIELD: str(r.get(QUESTION_FIELD, "")),
                GOLD_ANSWER_FIELD: str(r.get(GOLD_FIELD, "")),
                TEACHER_EXTRACTED_FIELD: str(r.get(TEACHER_PRED_FIELD, "")),
                TEACHER_RAW_PREDICTION_FIELD: str(r.get(TEACHER_RAW_FIELD, "")),
                STUDENT_EXTRACTED_FIELD: str(r.get(STUDENT_PRED_FIELD, "")),
                STUDENT_RAW_PREDICTION_FIELD: str(r.get(STUDENT_RAW_FIELD, "")),
                TEACHER_MATCH_FIELD: 1 if bool(r.get(TEACHER_OK_FIELD, False)) else 0,
                STUDENT_MATCH_FIELD: 1 if bool(r.get(STUDENT_OK_FIELD, False)) else 0,
                WINNER_LABEL_FIELD: str(r.get(WINNER_FIELD, "")),
            }
        )
    write_jsonl_rows(path, flat_rows)


def extract_score(result_json: Path, task_name: str) -> tuple[float | None, str | None]:
    payload = json.loads(result_json.read_text(encoding="utf-8"))
    results = payload.get("results", {})
    task = results.get(task_name, {})
    if not isinstance(task, dict):
        return None, None

    preferred = ("exact_match,marker-priority", "exact_match,flexible-extract", "exact_match")
    for key in preferred:
        val = task.get(key)
        if isinstance(val, (int, float)):
            return float(val), key

    fallback_key: str | None = None
    fallback_val: float | None = None
    for k, v in task.items():
        if "exact_match" in k and "stderr" not in k and isinstance(v, (int, float)):
            fallback_key = k
            fallback_val = float(v)
            break
    return fallback_val, fallback_key


def _load_reuse_summary(path_arg: str | None, role: str) -> dict[str, Any] | None:
    if not path_arg:
        return None
    p = Path(path_arg)
    if not p.exists():
        raise FileNotFoundError(f"reuse-{role} comparison.json not found: {p}")
    payload = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"Invalid reuse-{role} comparison payload (expected JSON object): {p}")
    return payload


def _path_from_summary(summary: dict[str, Any], key: str, role: str) -> Path | None:
    raw = summary.get(key)
    if not isinstance(raw, str) or not raw.strip():
        return None
    p = Path(raw)
    if p.exists():
        return p
    LOGGER.warning("Reuse-%s path missing for key=%s: %s", role, key, p)
    return None


def _count_jsonl_rows(path: Path) -> int:
    count = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                count += 1
    return count


def _reuse_dataset_jsonl(summary: dict[str, Any], out_jsonl: Path) -> tuple[int, str]:
    raw = summary.get("dataset_jsonl")
    if not isinstance(raw, str) or not raw.strip():
        raise RuntimeError("Reuse-dataset comparison is missing dataset_jsonl.")
    src = Path(raw)
    if not src.exists():
        raise FileNotFoundError(f"Reuse-dataset JSONL not found: {src}")
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, out_jsonl)
    return _count_jsonl_rows(out_jsonl), str(src)


def update_leaderboard(root_dir: Path, summary: dict[str, Any]) -> None:
    root_dir.mkdir(parents=True, exist_ok=True)
    lb_jsonl = root_dir / "leaderboard.jsonl"
    rows: list[dict[str, Any]] = []
    if lb_jsonl.exists():
        with lb_jsonl.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    rows.append(summary)

    with lb_jsonl.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    sortable = [
        r
        for r in rows
        if isinstance(r.get("teacher_exact_match"), (int, float)) and isinstance(r.get("student_exact_match"), (int, float))
    ]
    sortable.sort(key=lambda r: float(r.get("teacher_exact_match", 0.0)), reverse=True)

    md = []
    md.append("# lm-eval Leaderboard")
    md.append("")
    md.append("| Rank | Eval Name | Created (UTC) | Rows | Teacher EM | Student EM | Delta |")
    md.append("|---:|---|---|---:|---:|---:|---:|")
    for i, r in enumerate(sortable, start=1):
        md.append(
            f"| {i} | {r.get('eval_name')} | {r.get('created_at_utc')} | {r.get('num_rows')} | "
            f"{float(r.get('teacher_exact_match')):.6f} | {float(r.get('student_exact_match')):.6f} | "
            f"{float(r.get('delta_teacher_minus_student', 0.0)):.6f} |"
        )
    (root_dir / "leaderboard.md").write_text("\n".join(md), encoding="utf-8")

    cfg = {
        "schema_version": 1,
        "primary_metric": "teacher_exact_match",
        "secondary_metric": "delta_teacher_minus_student",
        "sort_order": "desc",
        "registry_file": str(lb_jsonl),
    }
    (root_dir / "leaderboard_config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")


def update_tracking_registry(root_dir: Path, summary: dict[str, Any]) -> None:
    """
    Append a stable run record and keep lightweight pointers for latest-per-stage.
    """
    root_dir.mkdir(parents=True, exist_ok=True)
    reg = root_dir / "eval_tracking_registry.jsonl"
    with reg.open("a", encoding="utf-8") as f:
        f.write(json.dumps(summary, ensure_ascii=False) + "\n")

    stage = slugify(str(summary.get("stage", "adhoc")))
    latest_dir = root_dir / "_latest_by_stage"
    latest_dir.mkdir(parents=True, exist_ok=True)
    (latest_dir / f"{stage}.txt").write_text(str(summary.get("eval_name", "")) + "\n", encoding="utf-8")


def resolve_model_ref(model_ref: str, local_only: bool) -> tuple[str, bool]:
    """
    Returns:
      - model_ref_for_lm_eval (absolute local path if directory exists, otherwise original string)
      - is_local_path
    """
    def infer_hf_repo_id(ref: str) -> str | None:
        # Common local cache folder name:
        #   models--ORG--NAME  -> ORG/NAME
        # Also supports full paths containing that segment.
        for part in Path(ref).expanduser().parts:
            if part.startswith("models--") and part.count("--") >= 2:
                body = part[len("models--") :]
                repo = body.replace("--", "/", 1)
                return repo if "/" in repo else None
        name = Path(ref).name
        if name.startswith("models--") and name.count("--") >= 2:
            body = name[len("models--") :]
            repo = body.replace("--", "/", 1)
            return repo if "/" in repo else None
        return None

    p = Path(model_ref).expanduser()
    if p.exists():
        p = p.resolve()
        # Handle Hugging Face cache-root layout:
        #   models--ORG--NAME/{blobs,refs,snapshots/<rev>/...}
        # Transformers expects the snapshot dir that contains config.json.
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
                snaps = sorted([d for d in snap_root.iterdir() if d.is_dir()], key=lambda d: d.stat().st_mtime, reverse=True)
                if snaps:
                    chosen = snaps[0]
            if chosen is None:
                raise FileNotFoundError(f"No snapshot directories found in HF cache path: {p}")
            p = chosen

        if p.is_dir() and not (p / "config.json").exists():
            raise FileNotFoundError(
                f"Local model directory does not contain config.json: {p}"
            )

        if p.is_dir():
            has_weights = any((p / n).exists() for n in ("model.safetensors", "pytorch_model.bin", "model.safetensors.index.json", "pytorch_model.bin.index.json"))
            if not has_weights:
                raise FileNotFoundError(
                    f"Local model directory found but no weights file present: {p}"
                )

        return str(p), True
    if local_only:
        raise FileNotFoundError(
            f"--local-models-only was set, but local model path was not found: {p}"
        )
    inferred_repo = infer_hf_repo_id(model_ref)
    if inferred_repo:
        LOGGER.info(
            "Local model path not found: %s. Falling back to Hugging Face repo: %s",
            model_ref,
            inferred_repo,
        )
        return inferred_repo, False
    return model_ref, False


def render_html_report(
    *,
    run_dir: Path,
    root_dir: Path,
    title: str,
    row_limit: int,
    per_eval_sample_limit: int,
) -> Path | None:
    renderer_path = Path(__file__).with_name("render_lm_eval_html.py")
    if not renderer_path.exists():
        LOGGER.warning("HTML renderer not found at %s. Skipping report.html generation.", renderer_path)
        return None
    spec = importlib.util.spec_from_file_location("render_lm_eval_html", renderer_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load html renderer module: {renderer_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    comp = run_dir / "comparison.json"
    current = module.load_json(comp)
    evals, samples_by_eval = module.collect_all_evals(
        root_dir,
        per_eval_sample_limit=max(0, int(per_eval_sample_limit)),
    )
    effective_row_limit = int(row_limit)
    if effective_row_limit <= 0:
        # Render all rows by using a very high cap for the client-side table slice.
        effective_row_limit = 10**9

    html_text = module.render_html(
        run_dir=run_dir,
        current=current,
        evals=evals,
        samples_by_eval=samples_by_eval,
        title=title,
        row_limit=effective_row_limit,
    )
    out = run_dir / "report.html"
    out.write_text(html_text, encoding="utf-8")
    return out


def main() -> None:
    configure_logging()
    args = parse_args()
    if args.resume_student_only and args.overwrite_output:
        raise ValueError("--resume-student-only cannot be combined with --overwrite-output.")
    if args.resume_student_only and args.reuse_student_comparison_json:
        raise ValueError("--resume-student-only cannot be combined with --reuse-student-comparison-json.")
    if args.resume_student_only and args.reuse_teacher_comparison_json:
        raise ValueError("--resume-student-only cannot be combined with --reuse-teacher-comparison-json.")
    if args.resume_student_only and args.reuse_dataset_jsonl_from_comparison:
        raise ValueError("--resume-student-only cannot be combined with --reuse-dataset-jsonl-from-comparison.")
    if args.fast_mode:
        if int(args.render_html) != 0:
            LOGGER.info("Fast mode enabled: forcing --render-html=0")
        args.render_html = 0
    reused_teacher_summary = _load_reuse_summary(args.reuse_teacher_comparison_json, "teacher")
    reused_student_summary = _load_reuse_summary(args.reuse_student_comparison_json, "student")
    reused_dataset_summary = _load_reuse_summary(args.reuse_dataset_jsonl_from_comparison, "dataset")
    limit_cfg = resolve_limit_config(
        args.limit,
        allow_env_limit=bool(args.allow_env_limit),
    )
    if limit_cfg.get("ignored_env_limit"):
        LOGGER.warning(
            "Ignoring inherited env limit=%s because --allow-env-limit was not set.",
            limit_cfg.get("ignored_env_limit"),
        )
    log_state(
        "eval_limit_status",
        limit_active=int(bool(limit_cfg.get("active"))),
        limit_value=limit_cfg.get("value") or "none",
        limit_source=limit_cfg.get("source") or "none",
    )
    log_state("eval_run_start", stage=args.stage, experiment=args.experiment, split=args.split)
    ensure_lm_eval_installed()
    base_root = Path("newoutput") / "lm_eval"
    stage_slug = slugify(args.stage)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    eval_name_raw = args.eval_name or f"{stage_slug}-{ts}"
    eval_name = slugify(eval_name_raw)
    out_dir = Path(args.output_dir) if args.output_dir else (base_root / eval_name)
    eval_name_final = out_dir.name
    run_id = args.run_id or f"{stage_slug}-{ts}"
    log_state("output_resolved", out_dir=str(out_dir), eval_name=eval_name_final, run_id=run_id)
    if out_dir.exists() and any(out_dir.iterdir()) and args.overwrite_output:
        log_state("output_overwrite_start", out_dir=str(out_dir))
        for child in out_dir.iterdir():
            if child.is_dir():
                shutil.rmtree(child)
            else:
                child.unlink()
        log_state("output_overwrite_end", out_dir=str(out_dir))
    out_dir.mkdir(parents=True, exist_ok=True)

    task_slug = taskify(eval_name_final)
    task_name = f"gsm8k_local_{task_slug}"
    data_jsonl = out_dir / f"{task_name}_{args.split}.jsonl"
    dataset_arrow_str: str
    log_state(
        "dataset_prepare_start",
        dataset_path=args.dataset_path,
        dataset_jsonl=args.dataset_jsonl or "none",
        split=args.split,
    )
    if reused_dataset_summary is not None:
        n_rows, reused_dataset_jsonl = _reuse_dataset_jsonl(reused_dataset_summary, data_jsonl)
        dataset_arrow_str = str(reused_dataset_summary.get("dataset_arrow") or f"reused-jsonl:{reused_dataset_jsonl}")
        log_state(
            "dataset_prepare_end",
            source="reused-comparison-jsonl",
            rows=n_rows,
            jsonl=str(data_jsonl),
            reused_from=str(args.reuse_dataset_jsonl_from_comparison),
        )
    elif args.dataset_jsonl:
        src_jsonl = Path(str(args.dataset_jsonl))
        if not src_jsonl.exists():
            raise FileNotFoundError(f"--dataset-jsonl file not found: {src_jsonl}")
        n_rows = copy_jsonl_dataset(src_jsonl, data_jsonl, args.max_samples)
        dataset_arrow_str = f"jsonl://{src_jsonl}"
        log_state("dataset_prepare_end", source="jsonl", rows=n_rows, jsonl=str(data_jsonl))
    else:
        try:
            split_arrow = resolve_split_arrow(args.dataset_path, args.split)
            n_rows = to_jsonl_from_arrow(split_arrow, data_jsonl, args.max_samples)
            dataset_arrow_str = str(split_arrow)
            log_state("dataset_prepare_end", source="arrow", rows=n_rows, jsonl=str(data_jsonl))
        except FileNotFoundError:
            LOGGER.info(
                "Dataset path not found/unusable: %s. Falling back to Hugging Face dataset openai/gsm8k (main).",
                args.dataset_path,
            )
            n_rows = to_jsonl_from_hf(args.split, data_jsonl, args.max_samples)
            dataset_arrow_str = "hf://openai/gsm8k/main"
            log_state("dataset_prepare_end", source="hf", rows=n_rows, jsonl=str(data_jsonl))

    overlap_report_path = out_dir / "overlap_report.json"
    overlap_refs: list[tuple[str, Path]] = []
    for raw_ref in args.overlap_reference:
        name, path = parse_named_path(raw_ref)
        overlap_refs.append((name, path))
    overlap_report: dict[str, Any] | None = None
    if overlap_refs:
        eval_rows = read_sample_rows(data_jsonl, limit=None)
        overlap_report = evaluate_overlap_guardrails(
            eval_rows=eval_rows,
            references=overlap_refs,
            out_path=overlap_report_path,
            threshold=float(args.overlap_threshold),
            allow_overlap=bool(args.allow_overlap),
        )
        log_state(
            "overlap_check_done",
            report=str(overlap_report_path),
            violations=len(overlap_report.get("violations", [])),
            max_overlap_pct=f"{float(overlap_report.get('max_overlap_pct', 0.0)):.6f}",
        )

    effective_num_fewshot = min(int(args.num_fewshot), max(0, int(n_rows) - 1))
    if effective_num_fewshot != int(args.num_fewshot):
        LOGGER.warning(
            "Reducing num_fewshot from %s to %s because dataset rows=%s.",
            args.num_fewshot,
            effective_num_fewshot,
            n_rows,
        )

    task_yaml = out_dir / f"{task_name}.yaml"
    task_name = write_task_yaml(
        task_yaml,
        data_jsonl,
        effective_num_fewshot,
        task_name,
        fewshot_cot=bool(args.fewshot_cot),
    )
    log_state("task_yaml_written", task_name=task_name, task_yaml=str(task_yaml))

    if reused_teacher_summary is not None:
        teacher_ref = str(reused_teacher_summary.get("teacher_model") or args.teacher_model)
        teacher_is_local = bool(reused_teacher_summary.get("teacher_model_is_local_path", Path(teacher_ref).exists()))
    else:
        teacher_ref, teacher_is_local = resolve_model_ref(args.teacher_model, args.local_models_only)
    if reused_student_summary is not None:
        student_ref = str(reused_student_summary.get("student_model") or args.student_model)
        student_is_local = bool(reused_student_summary.get("student_model_is_local_path", Path(student_ref).exists()))
    else:
        student_ref, student_is_local = resolve_model_ref(args.student_model, args.local_models_only)
    log_state("model_refs_resolved", teacher_local=teacher_is_local, student_local=student_is_local)

    common = [
        "uv",
        "run",
        "python",
        "-m",
        "lm_eval",
        "run",
        "--model",
        "hf",
        "--tasks",
        task_name,
        "--include_path",
        str(out_dir),
        "--batch_size",
        str(args.batch_size),
        "--device",
        str(args.device),
        "--log_samples",
        "--gen_kwargs",
        f"do_sample=False,temperature=0.0,max_gen_toks={int(args.gen_max_toks)}",
    ]
    if bool(limit_cfg.get("active")) and limit_cfg.get("value") is not None:
        common.extend(["--limit", str(limit_cfg["value"])])

    teacher_out = out_dir / "teacher"
    student_out = out_dir / "student"

    teacher_model_args = [f"pretrained={teacher_ref}", "trust_remote_code=True", "dtype=auto"]
    student_model_args = [f"pretrained={student_ref}", "trust_remote_code=True", "dtype=auto"]
    if int(args.gpu_optimize_6gb) == 1 and str(args.device).startswith("cuda"):
        six_gb_profile = [
            "dtype=float16",
        ]
        teacher_model_args = _merge_model_arg_list(teacher_model_args, six_gb_profile)
        student_model_args = _merge_model_arg_list(student_model_args, six_gb_profile)
    if args.local_models_only and teacher_is_local:
        teacher_model_args.append("local_files_only=True")
    if args.local_models_only and student_is_local:
        student_model_args.append("local_files_only=True")

    teacher_cmd = common + [
        "--model_args",
        ",".join(teacher_model_args),
        "--output_path",
        str(teacher_out),
    ]
    student_cmd = common + [
        "--model_args",
        ",".join(student_model_args),
        "--output_path",
        str(student_out),
    ]

    if args.dry_run:
        if args.resume_student_only:
            LOGGER.info("DRY teacher: skipped (--resume-student-only)")
        elif reused_teacher_summary is not None:
            LOGGER.info("DRY teacher: skipped (--reuse-teacher-comparison-json=%s)", args.reuse_teacher_comparison_json)
        else:
            LOGGER.info("DRY teacher: %s", " ".join(teacher_cmd))
        if reused_student_summary is not None:
            LOGGER.info("DRY student: skipped (--reuse-student-comparison-json=%s)", args.reuse_student_comparison_json)
        else:
            LOGGER.info("DRY student: %s", " ".join(student_cmd))
        log_state("dry_run_end", teacher_cmd_len=len(teacher_cmd), student_cmd_len=len(student_cmd))
        return

    rtx_4050_env = os.environ.copy()
    rtx_4050_env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    if args.resume_student_only:
        probe_teacher_res, probe_teacher_samples = normalize_model_artifacts(teacher_out)
        if probe_teacher_res is None:
            raise FileNotFoundError(
                f"--resume-student-only requires existing teacher results under {teacher_out}."
            )
        log_state("teacher_eval_skipped_resume_student_only", output=str(teacher_out))
    elif reused_teacher_summary is not None:
        log_state("teacher_eval_skipped_reuse", comparison_json=str(args.reuse_teacher_comparison_json))
    else:
        teacher_eval_started = time.perf_counter()
        log_state("teacher_eval_start", output=str(teacher_out))
        try:
            run_cmd(teacher_cmd, Path.cwd(), retries=max(0, int(args.retry_count)), echo_subprocess=bool(args.echo_subprocess))
        except subprocess.CalledProcessError:
            fallback_gen_toks = max(16, int(args.teacher_fallback_max_gen_toks))
            fallback_cmd = _replace_flag_value(teacher_cmd, "--device", str(args.teacher_fallback_device))
            fallback_cmd = _replace_flag_value(
                fallback_cmd,
                "--gen_kwargs",
                f"do_sample=False,temperature=0.0,max_gen_toks={fallback_gen_toks}",
            )
            fallback_env: dict[str, str] | None = None
            if int(args.gpu_optimize_6gb) == 1 and str(args.teacher_fallback_device).startswith("cuda"):
                LOGGER.warning(
                    "Teacher lm-eval failed. Applying RTX 4050 6GB CUDA profile (fp16 + allocator tuning) before fallback."
                )
                fallback_cmd = _append_model_args(fallback_cmd, ["dtype=float16"])
                fallback_env = rtx_4050_env
            LOGGER.warning(
                "Teacher lm-eval failed after retries. Retrying with fallback device=%s max_gen_toks=%s.",
                args.teacher_fallback_device,
                fallback_gen_toks,
            )
            try:
                run_cmd(fallback_cmd, Path.cwd(), retries=0, env=fallback_env, echo_subprocess=bool(args.echo_subprocess))
            except subprocess.CalledProcessError:
                if int(args.gpu_optimize_6gb) != 1 or not str(args.teacher_fallback_device).startswith("cuda"):
                    raise
                offload_cmd = _append_model_args(
                    fallback_cmd,
                    [ 
                        "max_memory_per_gpu=5GiB",
                        "max_cpu_memory=48GiB",
                    ],
                )
                LOGGER.warning("Teacher CUDA retry still failed. Retrying with CUDA+CPU offload profile.")
                run_cmd(offload_cmd, Path.cwd(), retries=0, env=rtx_4050_env, echo_subprocess=bool(args.echo_subprocess))
        log_state("teacher_eval_end", output=str(teacher_out), elapsed_s=f"{(time.perf_counter() - teacher_eval_started):.2f}")

    if reused_student_summary is not None:
        log_state("student_eval_skipped_reuse", comparison_json=str(args.reuse_student_comparison_json))
    else:
        student_eval_started = time.perf_counter()
        log_state("student_eval_start", output=str(student_out))
        try:
            run_cmd(student_cmd, Path.cwd(), retries=max(0, int(args.retry_count)), echo_subprocess=bool(args.echo_subprocess))
        except subprocess.CalledProcessError:
            fallback_gen_toks = max(16, int(args.student_fallback_max_gen_toks))
            fallback_cmd = _replace_flag_value(student_cmd, "--device", str(args.student_fallback_device))
            fallback_cmd = _replace_flag_value(
                fallback_cmd,
                "--gen_kwargs",
                f"do_sample=False,temperature=0.0,max_gen_toks={fallback_gen_toks}",
            )
            fallback_env: dict[str, str] | None = None
            if int(args.gpu_optimize_6gb) == 1 and str(args.student_fallback_device).startswith("cuda"):
                LOGGER.warning(
                    "Student lm-eval failed. Applying RTX 4050 6GB CUDA profile (fp16 + allocator tuning) before fallback."
                )
                fallback_cmd = _append_model_args(fallback_cmd, ["dtype=float16"])
                fallback_env = rtx_4050_env
            LOGGER.warning(
                "Student lm-eval failed after retries. Retrying with fallback device=%s max_gen_toks=%s.",
                args.student_fallback_device,
                fallback_gen_toks,
            )
            try:
                run_cmd(fallback_cmd, Path.cwd(), retries=0, env=fallback_env, echo_subprocess=bool(args.echo_subprocess))
            except subprocess.CalledProcessError:
                if int(args.gpu_optimize_6gb) != 1 or not str(args.student_fallback_device).startswith("cuda"):
                    raise
                offload_cmd = _append_model_args(
                    fallback_cmd,
                    [
                        "parallelize=True",
                        "max_memory_per_gpu=5GiB",
                        "max_cpu_memory=48GiB",
                    ],
                )
                LOGGER.warning("Student CUDA retry still failed. Retrying with CUDA+CPU offload profile.")
                run_cmd(offload_cmd, Path.cwd(), retries=0, env=rtx_4050_env, echo_subprocess=bool(args.echo_subprocess))
        log_state("student_eval_end", output=str(student_out), elapsed_s=f"{(time.perf_counter() - student_eval_started):.2f}")

    log_state(
        "artifact_normalize_start",
        teacher_out=str(teacher_out),
        student_out=str(student_out),
        teacher_reused=bool(reused_teacher_summary is not None),
        student_reused=bool(reused_student_summary is not None),
    )
    if reused_teacher_summary is not None:
        teacher_res = _path_from_summary(reused_teacher_summary, "teacher_results_json", "teacher")
        teacher_samples = _path_from_summary(reused_teacher_summary, "teacher_samples_jsonl", "teacher")
    else:
        teacher_res, teacher_samples = normalize_model_artifacts(teacher_out)
    if reused_student_summary is not None:
        student_res = _path_from_summary(reused_student_summary, "student_results_json", "student")
        student_samples = _path_from_summary(reused_student_summary, "student_samples_jsonl", "student")
    else:
        student_res, student_samples = normalize_model_artifacts(student_out)
    log_state(
        "artifact_normalize_end",
        teacher_results=str(teacher_res) if teacher_res else "none",
        student_results=str(student_res) if student_res else "none",
    )

    if reused_teacher_summary is not None:
        lm_t_score_raw = reused_teacher_summary.get("teacher_lm_eval_exact_match")
        lm_t_score = float(lm_t_score_raw) if isinstance(lm_t_score_raw, (int, float)) else None
        lm_t_metric_key = reused_teacher_summary.get("teacher_lm_eval_metric_key")
        if lm_t_metric_key is not None:
            lm_t_metric_key = str(lm_t_metric_key)
    else:
        lm_t_score, lm_t_metric_key = extract_score(teacher_res, task_name) if teacher_res else (None, None)
    if reused_student_summary is not None:
        lm_s_score_raw = reused_student_summary.get("student_lm_eval_exact_match")
        lm_s_score = float(lm_s_score_raw) if isinstance(lm_s_score_raw, (int, float)) else None
        lm_s_metric_key = reused_student_summary.get("student_lm_eval_metric_key")
        if lm_s_metric_key is not None:
            lm_s_metric_key = str(lm_s_metric_key)
    else:
        lm_s_score, lm_s_metric_key = extract_score(student_res, task_name) if student_res else (None, None)

    sample_cmp = build_sample_comparison(
        teacher_samples,
        student_samples,
        limit=int(args.sample_comparison_limit),
    )
    examples_rows = sample_cmp.get(EXAMPLES_KEY, [])
    if not isinstance(examples_rows, list):
        examples_rows = []
    bucket_metrics = compute_bucket_metrics(examples_rows)
    teacher_extract_fail_count = sum(
        1 for r in examples_rows if not str(r.get(TEACHER_PRED_FIELD, "") or "").strip()
    )
    student_extract_fail_count = sum(
        1 for r in examples_rows if not str(r.get(STUDENT_PRED_FIELD, "") or "").strip()
    )
    loaded_rows = int(sample_cmp.get("loaded_rows", 0) or 0)
    expected_rows = int(args.expected_rows) if int(args.expected_rows) > 0 else int(n_rows)
    partial_eval = loaded_rows < expected_rows
    likely_source = infer_partial_eval_source(
        limit_cfg=limit_cfg,
        sample_comparison_limit=int(args.sample_comparison_limit),
        reused_teacher=bool(reused_teacher_summary is not None),
        reused_student=bool(reused_student_summary is not None),
    )
    integrity = build_eval_integrity_status(
        expected_rows=expected_rows,
        loaded_rows=loaded_rows,
        allow_partial_eval=bool(args.allow_partial_eval),
        likely_source=likely_source,
        limit_cfg=limit_cfg,
    )
    log_state("sample_comparison_built", loaded_rows=sample_cmp.get("loaded_rows", 0))
    # Canonical score protocol: marker-priority extraction from raw generations.
    # Use it as the leaderboard metric only when comparison loaded the full dataset.
    use_local_protocol = loaded_rows == expected_rows and expected_rows > 0
    if use_local_protocol:
        t_score = sample_cmp.get("teacher_accuracy")
        s_score = sample_cmp.get("student_accuracy")
        t_metric_key = "exact_match,marker-priority(local)"
        s_metric_key = "exact_match,marker-priority(local)"
    else:
        t_score = None
        s_score = None
        t_metric_key = None
        s_metric_key = None
    if t_score is None:
        t_score = lm_t_score
        t_metric_key = lm_t_metric_key
    if reused_teacher_summary is not None and t_score is None:
        reused_teacher_score = reused_teacher_summary.get("teacher_exact_match")
        if isinstance(reused_teacher_score, (int, float)):
            t_score = float(reused_teacher_score)
            t_metric_key = str(reused_teacher_summary.get("teacher_metric_key") or "reused")
    if s_score is None:
        s_score = lm_s_score
        s_metric_key = lm_s_metric_key
    if reused_student_summary is not None and s_score is None:
        reused_student_score = reused_student_summary.get("student_exact_match")
        if isinstance(reused_student_score, (int, float)):
            s_score = float(reused_student_score)
            s_metric_key = str(reused_student_summary.get("student_metric_key") or "reused")

    sample_columns_jsonl = out_dir / "sample_columns.jsonl"
    write_sample_columns_jsonl(sample_columns_jsonl, sample_cmp)
    log_state("sample_columns_written", path=str(sample_columns_jsonl))

    summary = {
        "created_at_utc": utc_now(),
        "run_id": run_id,
        "stage": args.stage,
        "experiment": args.experiment,
        "parent_eval": args.parent_eval,
        "notes": args.notes,
        "eval_name": eval_name_final,
        "task_name": task_name,
        "dataset_arrow": dataset_arrow_str,
        "dataset_jsonl": str(data_jsonl),
        "num_rows": n_rows,
        "expected_rows": expected_rows,
        "teacher_model": teacher_ref,
        "student_model": student_ref,
        "teacher_model_is_local_path": teacher_is_local,
        "student_model_is_local_path": student_is_local,
        "local_models_only": bool(args.local_models_only),
        "fast_mode": bool(args.fast_mode),
        "teacher_reused_from_comparison_json": str(args.reuse_teacher_comparison_json) if reused_teacher_summary is not None else None,
        "student_reused_from_comparison_json": str(args.reuse_student_comparison_json) if reused_student_summary is not None else None,
        "dataset_reused_from_comparison_json": str(args.reuse_dataset_jsonl_from_comparison) if reused_dataset_summary is not None else None,
        "teacher_results_json": str(teacher_res) if teacher_res else None,
        "student_results_json": str(student_res) if student_res else None,
        "teacher_samples_jsonl": str(teacher_samples) if teacher_samples else None,
        "student_samples_jsonl": str(student_samples) if student_samples else None,
        "teacher_metric_key": t_metric_key,
        "student_metric_key": s_metric_key,
        "limit_active": bool(limit_cfg.get("active")),
        "limit_value": limit_cfg.get("value"),
        "limit_source": limit_cfg.get("source"),
        "overlap_threshold": float(args.overlap_threshold),
        "allow_overlap": bool(args.allow_overlap),
        "overlap_report_json": str(overlap_report_path) if overlap_refs else None,
        "overlap_violations": (
            len(overlap_report.get("violations", []))
            if isinstance(overlap_report, dict)
            else 0
        ),
        "teacher_lm_eval_exact_match": lm_t_score,
        "student_lm_eval_exact_match": lm_s_score,
        "teacher_lm_eval_metric_key": lm_t_metric_key,
        "student_lm_eval_metric_key": lm_s_metric_key,
        "scoring_protocol": "marker-priority(local): The answer is -> #### -> last number",
        "uses_local_scoring_protocol": use_local_protocol,
        "teacher_exact_match": t_score,
        "student_exact_match": s_score,
        "delta_teacher_minus_student": (t_score - s_score) if (t_score is not None and s_score is not None) else None,
        "teacher_output": str(teacher_out),
        "student_output": str(student_out),
        "eval_integrity": integrity,
        "invalid_run": bool(partial_eval),
        "bucket_metrics": bucket_metrics,
        "extraction_failures": {
            "teacher_extract_fail_count": teacher_extract_fail_count,
            "student_extract_fail_count": student_extract_fail_count,
            "teacher_extract_fail_rate": (
                (teacher_extract_fail_count / loaded_rows) if loaded_rows > 0 else None
            ),
            "student_extract_fail_rate": (
                (student_extract_fail_count / loaded_rows) if loaded_rows > 0 else None
            ),
        },
        SAMPLE_COMPARISON_KEY: sample_cmp,
        "sample_columns_jsonl": str(sample_columns_jsonl),
    }
    if partial_eval and not bool(args.allow_partial_eval):
        invalid_msg = build_partial_eval_error_message(integrity)
        invalid_payload = {
            "created_at_utc": summary.get("created_at_utc"),
            "eval_name": eval_name_final,
            "invalid_run": True,
            "error": invalid_msg,
            "eval_integrity": integrity,
            "limit_status": {
                "active": bool(limit_cfg.get("active")),
                "value": limit_cfg.get("value"),
                "source": limit_cfg.get("source"),
            },
            "dataset_jsonl": str(data_jsonl),
            "overlap_report_json": str(overlap_report_path) if overlap_refs else None,
        }
        invalid_path = out_dir / "invalid_eval.json"
        invalid_path.write_text(json.dumps(invalid_payload, indent=2), encoding="utf-8")
        LOGGER.error(invalid_msg)
        LOGGER.error("Invalid eval artifact written: %s", invalid_path)
        raise RuntimeError(invalid_msg)
    (out_dir / "comparison.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (out_dir / "eval_metadata.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "stage": args.stage,
                "experiment": args.experiment,
                "parent_eval": args.parent_eval,
                "notes": args.notes,
                "created_at_utc": summary["created_at_utc"],
                "eval_name": eval_name_final,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    log_state("summary_written", comparison_json=str(out_dir / "comparison.json"), eval_metadata=str(out_dir / "eval_metadata.json"))
    if bool(summary.get("invalid_run")):
        LOGGER.warning("Skipping leaderboard/tracking update for invalid partial run: %s", eval_name_final)
    else:
        update_leaderboard(base_root, summary)
        update_tracking_registry(base_root, summary)
        log_state("tracking_updated", leaderboard=str(base_root / "leaderboard.jsonl"), registry=str(base_root / "eval_tracking_registry.jsonl"))

    lines = [
        "# Custom lm-eval GSM8K Report",
        "",
        f"- Generated (UTC): `{summary['created_at_utc']}`",
        f"- Run ID: `{run_id}`",
        f"- Stage: `{args.stage}`",
        f"- Experiment: `{args.experiment}`",
        f"- Parent Eval: `{args.parent_eval}`",
        f"- Task: `{task_name}`",
        f"- Dataset arrow: `{dataset_arrow_str}`",
        f"- Dataset jsonl: `{data_jsonl}`",
        f"- Rows: `{n_rows}`",
        f"- Fast mode: `{bool(args.fast_mode)}`",
        f"- Limit active: `{bool(limit_cfg.get('active'))}` (source: `{limit_cfg.get('source')}`, value: `{limit_cfg.get('value')}`)",
        f"- Teacher reused: `{args.reuse_teacher_comparison_json}`",
        f"- Student reused: `{args.reuse_student_comparison_json}`",
        f"- Dataset reused: `{args.reuse_dataset_jsonl_from_comparison}`",
        f"- Overlap report: `{overlap_report_path if overlap_refs else 'none'}`",
        f"- Overlap violations: `{len(overlap_report.get('violations', [])) if isinstance(overlap_report, dict) else 0}`",
        f"- Invalid run: `{bool(summary.get('invalid_run'))}`",
        f"- Integrity expected rows: `{expected_rows}`",
        f"- Integrity loaded rows: `{loaded_rows}`",
        "",
        "## Scores",
        "",
        f"- Teacher (`{args.teacher_model}`): `{t_score}` (metric: `{t_metric_key}`)",
        f"- Student (`{args.student_model}`): `{s_score}` (metric: `{s_metric_key}`)",
        f"- Teacher lm-eval metric: `{lm_t_score}` (`{lm_t_metric_key}`)",
        f"- Student lm-eval metric: `{lm_s_score}` (`{lm_s_metric_key}`)",
        f"- Delta (teacher - student): `{summary['delta_teacher_minus_student']}`",
        "",
        "## Sample Head-to-Head (first loaded rows)",
        "",
        f"- Loaded rows: `{sample_cmp['loaded_rows']}`",
        f"- Teacher wins: `{sample_cmp['teacher_wins']}`",
        f"- Student wins: `{sample_cmp['student_wins']}`",
        f"- Both correct: `{sample_cmp['both_correct']}`",
        f"- Both wrong: `{sample_cmp['both_wrong']}`",
        f"- Teacher accuracy (loaded rows): `{sample_cmp['teacher_accuracy']}`",
        f"- Student accuracy (loaded rows): `{sample_cmp['student_accuracy']}`",
        "",
        "## Bucket Metrics",
        "",
        f"- {BUCKET_TEACHER_CORRECT_STUDENT_WRONG}: `{bucket_metrics.get('bucket_counts', {}).get(BUCKET_TEACHER_CORRECT_STUDENT_WRONG, 0)}`",
        f"- {BUCKET_BOTH_CORRECT}: `{bucket_metrics.get('bucket_counts', {}).get(BUCKET_BOTH_CORRECT, 0)}`",
        f"- {BUCKET_BOTH_WRONG}: `{bucket_metrics.get('bucket_counts', {}).get(BUCKET_BOTH_WRONG, 0)}`",
        f"- {BUCKET_TEACHER_WRONG_STUDENT_CORRECT}: `{bucket_metrics.get('bucket_counts', {}).get(BUCKET_TEACHER_WRONG_STUDENT_CORRECT, 0)}`",
        f"- Recovery count (teacher correct / student wrong): `{bucket_metrics.get('recovery_count_teacher_correct_student_wrong')}`",
        f"- Damage count (both correct): `{bucket_metrics.get('damage_count_both_correct')}`",
        f"- Damage count (teacher wrong / student correct): `{bucket_metrics.get('damage_count_teacher_wrong_student_correct')}`",
        "",
        "## Outputs",
        "",
        f"- Teacher results: `{teacher_res}`",
        f"- Student results: `{student_res}`",
        f"- Teacher samples: `{teacher_samples}`",
        f"- Student samples: `{student_samples}`",
        f"- Comparison: `{out_dir / 'comparison.json'}`",
        f"- Sample columns: `{sample_columns_jsonl}`",
    ]
    (out_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")
    log_state("report_written", report_md=str(out_dir / "report.md"))

    html_out: Path | None = None
    if int(args.render_html) == 1:
        log_state("html_render_start", run_dir=str(out_dir))
        html_out = render_html_report(
            run_dir=out_dir,
            root_dir=base_root,
            title=args.html_title,
            row_limit=args.html_limit,
            per_eval_sample_limit=args.html_per_eval_sample_limit,
        )
        log_state("html_render_end", html=str(html_out) if html_out is not None else "none")

    LOGGER.info("DONE out_dir=%s", out_dir)
    LOGGER.info("DONE comparison=%s", out_dir / "comparison.json")
    LOGGER.info("DONE report=%s", out_dir / "report.md")
    if html_out is not None:
        LOGGER.info("DONE html=%s", html_out)
    LOGGER.info("DONE leaderboard=%s", base_root / "leaderboard.md")
    LOGGER.info("DONE tracking_registry=%s", base_root / "eval_tracking_registry.jsonl")
    log_state("eval_run_end", eval_name=eval_name_final, out_dir=str(out_dir))


if __name__ == "__main__":
    main()
