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
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from datasets import Dataset, load_dataset

DEFAULT_STUDENT_MODEL = "/media/prasanna/716F26140AED9B67/models/models--Qwen--Qwen2-0.5B-Instruct"
DEFAULT_TEACHER_MODEL = "/media/prasanna/716F26140AED9B67/models/models--Qwen--Qwen2-Math-1.5B-Instruct"
DEFAULT_GSM8K_PATH = "/media/prasanna/716F26140AED9B67/datasets/GSM8K"
LOGGER = logging.getLogger("run_lm_eval_custom")
COMMAND_TAIL_LINES = 25
HEARTBEAT_SECONDS = 30
ECHO_SUBPROCESS = False


def log_state(state: str, **fields: Any) -> None:
    payload = " ".join(f"{k}={fields[k]}" for k in sorted(fields))
    if payload:
        LOGGER.info("STATE | %s | %s", state, payload)
    else:
        LOGGER.info("STATE | %s", state)


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
        force=True,
    )


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run custom lm-eval benchmark on local GSM8K-style dataset")
    p.add_argument("--dataset-path", type=str, default=DEFAULT_GSM8K_PATH)
    p.add_argument("--split", choices=["train", "test"], default="test")
    p.add_argument("--max-samples", type=int, default=0)
    p.add_argument("--student-model", type=str, default=DEFAULT_STUDENT_MODEL)
    p.add_argument("--teacher-model", type=str, default=DEFAULT_TEACHER_MODEL)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--batch-size", type=str, default="1")
    p.add_argument("--limit", type=str, default=None, help="lm-eval limit, e.g. 100 or 0.1")
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
            q = str(row.get("question", "")).strip()
            a = str(row.get("answer", "")).strip()
            if not q or not a:
                continue
            f.write(json.dumps({"question": q, "answer": a}, ensure_ascii=False) + "\n")
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
            q = str(row.get("question", "")).strip()
            a = str(row.get("answer", "")).strip()
            if not q or not a:
                continue
            f.write(json.dumps({"question": q, "answer": a}, ensure_ascii=False) + "\n")
            count += 1
            if max_samples > 0 and count >= max_samples:
                break
    return count


def write_task_yaml(
    task_path: Path,
    data_jsonl: Path,
    num_fewshot: int,
    task_name: str,
    fewshot_cot: bool,
) -> str:
    doc_to_target = "{{answer}}" if fewshot_cot else "{{answer.split('####')[-1].strip()}}"
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
  Q: {{{{question}}}}

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


def run_cmd(cmd: list[str], cwd: Path, retries: int = 0, env: dict[str, str] | None = None) -> None:
    LOGGER.info("CMD: %s", " ".join(cmd))
    last_exc: subprocess.CalledProcessError | None = None
    for attempt in range(retries + 1):
        start = time.perf_counter()
        tail: list[str] = []
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
                assert proc.stdout is not None
                for line in proc.stdout:
                    msg = line.rstrip("\n")
                    tail.append(msg)
                    if len(tail) > COMMAND_TAIL_LINES:
                        del tail[:-COMMAND_TAIL_LINES]
                    if ECHO_SUBPROCESS:
                        print(msg)

            t = threading.Thread(target=_reader, daemon=True)
            t.start()
            while proc.poll() is None:
                time.sleep(HEARTBEAT_SECONDS)
                elapsed = time.perf_counter() - start
                last_line = tail[-1] if tail else "(no output yet)"
                LOGGER.info("Still running (%.1fs): %s", elapsed, last_line[:200])
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
    question = str(doc.get("question", sample_row.get("prompt", "")))
    gold = str(sample_row.get("target", doc.get("answer", "")))

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
    if not text:
        return ""
    marker = re.search(r"(?is)the answer is\s*(-?[0-9]+(?:\.[0-9]+)?(?:,[0-9]{3})*)", text)
    if marker:
        return marker.group(1).strip()
    hash_marker = re.search(r"####\s*(-?[0-9]+(?:\.[0-9]+)?(?:,[0-9]{3})*)", text)
    if hash_marker:
        return hash_marker.group(1).strip()
    nums = re.findall(r"-?[0-9]+(?:\.[0-9]+)?", text.replace(",", ""))
    return nums[-1] if nums else ""


def _norm_text(x: str) -> str:
    preferred = _extract_priority_answer(x)
    s = preferred if preferred else x.strip().lower()
    m = re.search(r"####\s*([^\n\r]+)", s)
    if m:
        s = m.group(1).strip()
    s = s.replace(",", "").replace("$", "")
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def _extract_num(x: str) -> str | None:
    nums = re.findall(r"-?[0-9]+(?:\.[0-9]+)?", x)
    return nums[-1] if nums else None


def _matches(pred: str, gold: str) -> bool:
    p = _norm_text(pred)
    g = _norm_text(gold)
    if not g:
        return False
    if p == g:
        return True
    pn = _extract_num(p)
    gn = _extract_num(g)
    return pn is not None and gn is not None and pn == gn


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
            "examples": [],
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
                "question": question,
                "gold": gold,
                "teacher_pred": tp_extracted,
                "student_pred": sp_extracted,
                "teacher_raw": tp_raw,
                "student_raw": sp_raw,
                "teacher_ok": t_ok,
                "student_ok": s_ok,
                "winner": winner,
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
        "examples": examples,
    }


def write_sample_columns_jsonl(path: Path, sample_cmp: dict[str, Any]) -> None:
    """
    Write a flattened, analysis-friendly samples table with stable columns.
    """
    rows = sample_cmp.get("examples", [])
    if not isinstance(rows, list):
        rows = []
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            row = {
                "question": str(r.get("question", "")),
                "gold_answer": str(r.get("gold", "")),
                "teacher_extracted": str(r.get("teacher_pred", "")),
                "teacher_raw_prediction": str(r.get("teacher_raw", "")),
                "student": str(r.get("student_pred", "")),
                "student_extracted": str(r.get("student_pred", "")),
                "student_raw_prediction": str(r.get("student_raw", "")),
                "teacher_accuracy": 1 if bool(r.get("teacher_ok", False)) else 0,
                "student_accuracy": 1 if bool(r.get("student_ok", False)) else 0,
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


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
    inferred_repo = infer_hf_repo_id(model_ref)
    if inferred_repo:
        LOGGER.info(
            "Local model path not found: %s. Falling back to Hugging Face repo: %s",
            model_ref,
            inferred_repo,
        )
        return inferred_repo, False
    if local_only:
        LOGGER.warning(
            "--local-models-only was set, but local model path was not found: %s. Falling back to remote model resolution.",
            model_ref,
        )
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
    log_state("dataset_prepare_start", dataset_path=args.dataset_path, split=args.split)
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

    teacher_ref, teacher_is_local = resolve_model_ref(args.teacher_model, args.local_models_only)
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
    if args.limit:
        common.extend(["--limit", str(args.limit)])

    teacher_out = out_dir / "teacher"
    student_out = out_dir / "student"

    teacher_model_args = [f"pretrained={teacher_ref}", "trust_remote_code=True", "dtype=auto"]
    student_model_args = [f"pretrained={student_ref}", "trust_remote_code=True", "dtype=auto"]
    if int(args.gpu_optimize_6gb) == 1 and str(args.device).startswith("cuda"):
        offload_dir = out_dir / "offload"
        offload_dir.mkdir(parents=True, exist_ok=True)
        six_gb_profile = [
            "dtype=float16",
            "parallelize=True",
            "max_memory_per_gpu=5GiB",
            "max_cpu_memory=48GiB",
            f"offload_folder={offload_dir}",
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
        LOGGER.info("DRY teacher: %s", " ".join(teacher_cmd))
        LOGGER.info("DRY student: %s", " ".join(student_cmd))
        log_state("dry_run_end", teacher_cmd_len=len(teacher_cmd), student_cmd_len=len(student_cmd))
        return

    rtx_4050_env = os.environ.copy()
    rtx_4050_env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    teacher_eval_started = time.perf_counter()
    log_state("teacher_eval_start", output=str(teacher_out))
    try:
        run_cmd(teacher_cmd, Path.cwd(), retries=max(0, int(args.retry_count)))
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
            run_cmd(fallback_cmd, Path.cwd(), retries=0, env=fallback_env)
        except subprocess.CalledProcessError:
            if int(args.gpu_optimize_6gb) != 1 or not str(args.teacher_fallback_device).startswith("cuda"):
                raise
            offload_cmd = _append_model_args(
                fallback_cmd,
                [
                    "parallelize=True",
                    "max_memory_per_gpu=5GiB",
                    "max_cpu_memory=48GiB",
                ],
            )
            LOGGER.warning("Teacher CUDA retry still failed. Retrying with CUDA+CPU offload profile.")
            run_cmd(offload_cmd, Path.cwd(), retries=0, env=rtx_4050_env)
    log_state("teacher_eval_end", output=str(teacher_out), elapsed_s=f"{(time.perf_counter() - teacher_eval_started):.2f}")

    student_eval_started = time.perf_counter()
    log_state("student_eval_start", output=str(student_out))
    try:
        run_cmd(student_cmd, Path.cwd(), retries=max(0, int(args.retry_count)))
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
            run_cmd(fallback_cmd, Path.cwd(), retries=0, env=fallback_env)
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
            run_cmd(offload_cmd, Path.cwd(), retries=0, env=rtx_4050_env)
    log_state("student_eval_end", output=str(student_out), elapsed_s=f"{(time.perf_counter() - student_eval_started):.2f}")

    log_state("artifact_normalize_start", teacher_out=str(teacher_out), student_out=str(student_out))
    teacher_res, teacher_samples = normalize_model_artifacts(teacher_out)
    student_res, student_samples = normalize_model_artifacts(student_out)
    log_state(
        "artifact_normalize_end",
        teacher_results=str(teacher_res) if teacher_res else "none",
        student_results=str(student_res) if student_res else "none",
    )

    lm_t_score, lm_t_metric_key = extract_score(teacher_res, task_name) if teacher_res else (None, None)
    lm_s_score, lm_s_metric_key = extract_score(student_res, task_name) if student_res else (None, None)

    sample_cmp = build_sample_comparison(
        teacher_samples,
        student_samples,
        limit=int(args.sample_comparison_limit),
    )
    log_state("sample_comparison_built", loaded_rows=sample_cmp.get("loaded_rows", 0))
    # Canonical score protocol: marker-priority extraction from raw generations.
    # Use it as the leaderboard metric only when comparison loaded the full dataset.
    use_local_protocol = int(sample_cmp.get("loaded_rows", 0)) == int(n_rows) and int(n_rows) > 0
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
    if s_score is None:
        s_score = lm_s_score
        s_metric_key = lm_s_metric_key

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
        "teacher_model": teacher_ref,
        "student_model": student_ref,
        "teacher_model_is_local_path": teacher_is_local,
        "student_model_is_local_path": student_is_local,
        "local_models_only": bool(args.local_models_only),
        "teacher_results_json": str(teacher_res) if teacher_res else None,
        "student_results_json": str(student_res) if student_res else None,
        "teacher_samples_jsonl": str(teacher_samples) if teacher_samples else None,
        "student_samples_jsonl": str(student_samples) if student_samples else None,
        "teacher_metric_key": t_metric_key,
        "student_metric_key": s_metric_key,
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
        "sample_comparison": sample_cmp,
        "sample_columns_jsonl": str(sample_columns_jsonl),
    }
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
