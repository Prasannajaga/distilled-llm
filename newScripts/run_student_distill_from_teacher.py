#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
import random
import re
import shlex
import subprocess
import sys
import threading
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

LOGGER = logging.getLogger("student_distill")
TAIL_LINES = 25
HEARTBEAT_SECONDS = 30
DEFAULT_DATASET_PATH = "/media/prasanna/716F26140AED9B67/datasets/GSM8K"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def slugify(name: str) -> str:
    s = name.strip().lower()
    s = re.sub(r"[^a-z0-9._-]+", "-", s)
    s = re.sub(r"-{2,}", "-", s).strip("-")
    return s or "run"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Run controlled distillation phase: fixed teacher, filtered teacher-correct/student-wrong "
            "subset, target-format ablations, winner selection, and pure-vs-mixed comparison."
        )
    )
    p.add_argument("--summary-json", type=str, default=None, help="Optional teacher summary JSON with best_run fields.")
    p.add_argument("--teacher-model", type=str, default=None, help="Pinned teacher checkpoint/model path.")
    p.add_argument(
        "--comparison-json",
        type=str,
        default=None,
        help="comparison.json that contains sample_comparison/examples for filtering.",
    )

    p.add_argument("--experiment-name", type=str, default="distill-format-ablation")
    p.add_argument("--pipeline-run-id", type=str, default="")
    p.add_argument("--phase-name", type=str, default="next-distill-phase")
    p.add_argument("--parent-eval", type=str, default=None)
    p.add_argument("--output-root", type=str, default="newoutput/distill")

    p.add_argument("--max-filtered-rows", type=int, default=0, help="<=0 means all filtered rows.")
    p.add_argument("--max-answer-chars", type=int, default=2000)
    p.add_argument("--short-rationale-max-sentences", type=int, default=3)
    p.add_argument("--short-rationale-max-chars", type=int, default=360)

    p.add_argument("--student-base-model", type=str, default="")
    p.add_argument("--student-epochs", type=float, default=0.75)
    p.add_argument("--student-lr", type=float, default=8e-5)
    p.add_argument("--student-batch-size", type=int, default=4)
    p.add_argument("--student-grad-accum", type=int, default=4)
    p.add_argument("--student-max-seq-length", type=int, default=2048)
    p.add_argument("--student-max-train-samples", type=int, default=0)
    p.add_argument("--student-max-eval-samples", type=int, default=512)
    p.add_argument("--student-seed", type=int, default=42)
    p.add_argument("--train-val-ratio", type=float, default=0.1)
    p.add_argument("--split-seed", type=int, default=42)

    p.add_argument("--dataset-path", type=str, default=DEFAULT_DATASET_PATH, help="Continuity benchmark dataset path.")
    p.add_argument("--benchmark-split", choices=["train", "test"], default="test")
    p.add_argument("--benchmark-max-samples", type=int, default=500)
    p.add_argument("--heldout-dataset-path", type=str, default="")
    p.add_argument("--heldout-split", choices=["train", "test"], default="train")
    p.add_argument("--heldout-max-samples", type=int, default=500)

    p.add_argument("--num-fewshot", type=int, default=8)
    p.add_argument("--gen-max-toks", type=int, default=256)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--batch-size", type=str, default="1")
    p.add_argument("--limit", type=str, default=None)

    p.add_argument("--mixed-gold-ratio", type=float, default=0.30)
    p.add_argument("--mixed-seed", type=int, default=42)

    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
        force=True,
    )


def run_cmd(cmd: list[str], dry_run: bool) -> None:
    LOGGER.info("CMD: %s", shlex.join(cmd))
    if dry_run:
        LOGGER.info("Dry run mode: command skipped.")
        return
    start = time.perf_counter()
    last_heartbeat = start
    tail: deque[str] = deque(maxlen=TAIL_LINES)
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert proc.stdout is not None

    def _reader() -> None:
        assert proc.stdout is not None
        for line in proc.stdout:
            msg = line.rstrip("\n")
            tail.append(msg)

    t = threading.Thread(target=_reader, daemon=True)
    t.start()
    while proc.poll() is None:
        time.sleep(HEARTBEAT_SECONDS)
        now = time.perf_counter()
        if (now - last_heartbeat) >= HEARTBEAT_SECONDS:
            LOGGER.info("Still running (%.1fs): %s", now - start, tail[-1][:200] if tail else "(no output yet)")
            last_heartbeat = now
    t.join(timeout=5)
    rc = proc.wait()
    if rc != 0:
        if tail:
            LOGGER.error("Last %d output lines:\n%s", len(tail), "\n".join(tail))
        raise subprocess.CalledProcessError(rc, cmd)
    LOGGER.info("Command succeeded in %.2fs", time.perf_counter() - start)


def _extract_priority_answer(text: str) -> str:
    if not text:
        return ""
    marker = re.search(r"(?is)the answer is\s*(-?[0-9]+(?:\.[0-9]+)?(?:,[0-9]{3})*)", text)
    if marker:
        return marker.group(1).replace(",", "").strip()
    hash_marker = re.search(r"####\s*(-?[0-9]+(?:\.[0-9]+)?(?:,[0-9]{3})*)", text)
    if hash_marker:
        return hash_marker.group(1).replace(",", "").strip()
    nums = re.findall(r"-?[0-9]+(?:\.[0-9]+)?", text.replace(",", ""))
    return nums[-1] if nums else ""


def _normalize_ws(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def _strip_terminal_answer_markers(text: str) -> str:
    body = (text or "").strip()
    if not body:
        return ""
    body = re.sub(r"(?is)\n?\s*the answer is\s*-?[0-9]+(?:\.[0-9]+)?(?:,[0-9]{3})*\.?\s*$", "", body).strip()
    body = re.sub(r"(?is)\n?\s*####\s*-?[0-9]+(?:\.[0-9]+)?(?:,[0-9]{3})*\s*$", "", body).strip()
    return body


def _final_answer_marker(final_num: str) -> str:
    return f"The answer is {final_num}." if final_num else ""


def _build_short_rationale(body: str, marker: str, max_sentences: int, max_chars: int) -> str:
    if not body:
        return marker
    flat = _normalize_ws(body)
    chunks = [c.strip() for c in re.split(r"(?<=[.!?])\s+", flat) if c.strip()]
    if not chunks:
        chunks = [flat]
    selected: list[str] = []
    for c in chunks:
        selected.append(c)
        if max_sentences > 0 and len(selected) >= max_sentences:
            break
    short = " ".join(selected).strip()
    if max_chars > 0 and len(short) > max_chars:
        short = short[: max(1, max_chars)].rstrip()
        short = re.sub(r"\s+\S*$", "", short).strip() or short
    if marker and marker.lower() not in short.lower():
        short = f"{short}\n{marker}" if short else marker
    return short.strip()


def _render_answer(
    row: dict[str, Any],
    target_format: str,
    target_source: str,
    short_max_sentences: int,
    short_max_chars: int,
) -> str:
    teacher_raw = str(row.get("teacher_raw", "") or "")
    teacher_pred = str(row.get("teacher_pred", "") or "")
    gold = str(row.get("gold", "") or "")

    source_text = teacher_raw if target_source == "teacher" else gold
    final_num = (
        _extract_priority_answer(teacher_pred)
        or _extract_priority_answer(teacher_raw)
        or _extract_priority_answer(gold)
    )
    marker = _final_answer_marker(final_num)

    if target_format == "answer_only":
        return marker or _normalize_ws(source_text)

    body = _strip_terminal_answer_markers(source_text)
    if target_format == "short_rationale":
        return _build_short_rationale(body, marker, short_max_sentences, short_max_chars)

    if marker and marker.lower() not in body.lower():
        return f"{body}\n{marker}".strip() if body else marker
    return (body or marker).strip()


def _resolve_teacher_and_comparison(args: argparse.Namespace) -> tuple[str, str, str, str, str | None]:
    summary: dict[str, Any] = {}
    if args.summary_json:
        summary = json.loads(Path(args.summary_json).read_text(encoding="utf-8"))
    best = summary.get("best_run", {}) if isinstance(summary.get("best_run"), dict) else {}

    teacher_model = str(args.teacher_model or best.get("merged_dir") or "").strip()
    comparison_json = str(args.comparison_json or best.get("comparison_json") or "").strip()
    if not teacher_model:
        raise RuntimeError("Teacher model is required. Pass --teacher-model or --summary-json with best_run.merged_dir.")
    if not comparison_json:
        raise RuntimeError("Comparison JSON is required. Pass --comparison-json or --summary-json with best_run.comparison_json.")

    run_id = str(args.pipeline_run_id or summary.get("pipeline_run_id") or "").strip()
    if not run_id:
        run_id = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    experiment = str(args.experiment_name or summary.get("experiment_name") or "distill-format-ablation").strip()
    parent_eval = str(args.parent_eval or best.get("eval_name") or "").strip() or None
    return teacher_model, comparison_json, run_id, experiment, parent_eval


def _safe_int(x: Any, fallback: int) -> int:
    try:
        return int(x)
    except Exception:
        return fallback


def _compute_bucket_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    both_correct = 0
    both_wrong = 0
    teacher_only = 0
    student_only = 0
    for r in rows:
        t_ok = bool(r.get("teacher_ok", False))
        s_ok = bool(r.get("student_ok", False))
        if t_ok and s_ok:
            both_correct += 1
        elif t_ok and not s_ok:
            teacher_only += 1
        elif (not t_ok) and s_ok:
            student_only += 1
        else:
            both_wrong += 1
    return {
        "teacher correct / student wrong": teacher_only,
        "teacher wrong / student correct": student_only,
        "both correct": both_correct,
        "both wrong": both_wrong,
    }


def _filter_teacher_correct_student_wrong(rows: list[dict[str, Any]], max_rows: int) -> list[dict[str, Any]]:
    kept = [r for r in rows if bool(r.get("teacher_ok", False)) and (not bool(r.get("student_ok", False)))]
    if max_rows > 0:
        kept = kept[:max_rows]
    return kept


def _render_rows(
    filtered_rows: list[dict[str, Any]],
    target_format: str,
    target_source: str,
    short_max_sentences: int,
    short_max_chars: int,
    max_answer_chars: int,
) -> tuple[list[dict[str, Any]], list[int]]:
    out: list[dict[str, Any]] = []
    kept_ids: list[int] = []
    for i, row in enumerate(filtered_rows):
        question = str(row.get("question", "")).strip()
        answer = _render_answer(
            row,
            target_format=target_format,
            target_source=target_source,
            short_max_sentences=short_max_sentences,
            short_max_chars=short_max_chars,
        ).strip()
        if not question or not answer:
            continue
        if max_answer_chars > 0 and len(answer) > max_answer_chars:
            continue
        out.append({"question": question, "answer": answer})
        kept_ids.append(_safe_int(row.get("idx", i), i))
    return out, kept_ids


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _train_student(
    args: argparse.Namespace,
    train_jsonl: Path,
    run_tag: str,
    phase_slug: str,
    dry_run: bool,
) -> dict[str, Any]:
    run_slug = slugify(f"{phase_slug}-{run_tag}")
    lora_dir = Path("output") / f"student-distill-{run_slug}-lora"
    merged_dir = Path("output") / f"student-distill-{run_slug}-merged"

    cmd = [
        "uv",
        "run",
        "python",
        "newScripts/train_student_unsloth_distill.py",
        "--train-jsonl",
        str(train_jsonl),
        "--max-train-samples",
        str(args.student_max_train_samples),
        "--max-eval-samples",
        str(args.student_max_eval_samples),
        "--max-seq-length",
        str(args.student_max_seq_length),
        "--batch-size",
        str(args.student_batch_size),
        "--grad-accum",
        str(args.student_grad_accum),
        "--epochs",
        str(args.student_epochs),
        "--learning-rate",
        str(args.student_lr),
        "--seed",
        str(args.student_seed),
        "--val-split-ratio",
        str(args.train_val_ratio),
        "--split-seed",
        str(args.split_seed),
        "--output-dir",
        str(lora_dir),
        "--merged-output-dir",
        str(merged_dir),
        "--merge-16bit",
        "1",
    ]
    if args.student_base_model:
        cmd.extend(["--base-model", str(args.student_base_model)])

    run_cmd(cmd, dry_run)
    return {
        "run_tag": run_tag,
        "train_jsonl": str(train_jsonl),
        "train_cmd": shlex.join(cmd),
        "lora_dir": str(lora_dir),
        "merged_dir": str(merged_dir),
    }


def _parse_student_behavior_metrics(comparison_payload: dict[str, Any]) -> dict[str, Any]:
    rows = comparison_payload.get("sample_comparison", {}).get("examples", [])
    if not isinstance(rows, list):
        rows = []
    n = len(rows)
    if n <= 0:
        return {
            "loaded_rows": 0,
            "student_parse_fail_count": 0,
            "student_parse_fail_rate": None,
            "student_avg_output_chars": None,
        }

    parse_fails = 0
    total_chars = 0
    for row in rows:
        pred = str(row.get("student_pred", "") or "").strip()
        raw = str(row.get("student_raw", "") or "")
        if not pred:
            parse_fails += 1
        total_chars += len(raw)
    return {
        "loaded_rows": n,
        "student_parse_fail_count": parse_fails,
        "student_parse_fail_rate": (parse_fails / n) if n > 0 else None,
        "student_avg_output_chars": (total_chars / n) if n > 0 else None,
    }


def _eval_one_set(
    args: argparse.Namespace,
    teacher_model: str,
    student_model: str,
    eval_name: str,
    stage: str,
    dataset_path: str,
    split: str,
    max_samples: int,
    parent_eval: str | None,
    experiment: str,
    run_id: str,
    dry_run: bool,
) -> dict[str, Any]:
    cmd = [
        "uv",
        "run",
        "python",
        "newScripts/run_lm_eval_custom.py",
        "--dataset-path",
        str(dataset_path),
        "--split",
        split,
        "--max-samples",
        str(max_samples),
        "--teacher-model",
        teacher_model,
        "--student-model",
        student_model,
        "--device",
        args.device,
        "--batch-size",
        str(args.batch_size),
        "--num-fewshot",
        str(args.num_fewshot),
        "--gen-max-toks",
        str(args.gen_max_toks),
        "--fewshot-cot",
        "1",
        "--eval-name",
        eval_name,
        "--stage",
        stage,
        "--experiment",
        experiment,
        "--run-id",
        run_id,
        "--overwrite-output",
    ]
    if parent_eval:
        cmd.extend(["--parent-eval", parent_eval])
    if args.limit:
        cmd.extend(["--limit", str(args.limit)])

    run_cmd(cmd, dry_run)

    comparison_path = Path("newoutput") / "lm_eval" / eval_name / "comparison.json"
    if dry_run:
        return {
            "eval_name": eval_name,
            "stage": stage,
            "dataset_path": dataset_path,
            "split": split,
            "max_samples": max_samples,
            "comparison_json": str(comparison_path),
            "student_exact_match": None,
            "loaded_rows": 0,
            "student_parse_fail_count": 0,
            "student_parse_fail_rate": None,
            "student_avg_output_chars": None,
        }

    payload = _read_json(comparison_path)
    behavior = _parse_student_behavior_metrics(payload)
    em_raw = payload.get("student_exact_match")
    em = float(em_raw) if isinstance(em_raw, (int, float)) else None
    out = {
        "eval_name": eval_name,
        "stage": stage,
        "dataset_path": dataset_path,
        "split": split,
        "max_samples": max_samples,
        "comparison_json": str(comparison_path),
        "student_exact_match": em,
    }
    out.update(behavior)
    return out


def _evaluate_two_sets(
    args: argparse.Namespace,
    teacher_model: str,
    student_model: str,
    run_label: str,
    phase_slug: str,
    parent_eval: str | None,
    experiment: str,
    run_id: str,
    dry_run: bool,
) -> dict[str, Any]:
    run_slug = slugify(f"{phase_slug}-{run_label}")

    benchmark_eval_name = slugify(f"distill-benchmark-{run_slug}-{run_id}")
    benchmark = _eval_one_set(
        args,
        teacher_model=teacher_model,
        student_model=student_model,
        eval_name=benchmark_eval_name,
        stage="distill-format-benchmark",
        dataset_path=args.dataset_path,
        split=args.benchmark_split,
        max_samples=int(args.benchmark_max_samples),
        parent_eval=parent_eval,
        experiment=experiment,
        run_id=run_id,
        dry_run=dry_run,
    )

    heldout_dataset_path = args.heldout_dataset_path or args.dataset_path
    heldout_eval_name = slugify(f"distill-heldout-{run_slug}-{run_id}")
    heldout = _eval_one_set(
        args,
        teacher_model=teacher_model,
        student_model=student_model,
        eval_name=heldout_eval_name,
        stage="distill-format-heldout",
        dataset_path=heldout_dataset_path,
        split=args.heldout_split,
        max_samples=int(args.heldout_max_samples),
        parent_eval=parent_eval,
        experiment=experiment,
        run_id=run_id,
        dry_run=dry_run,
    )

    heldout_em = heldout.get("student_exact_match")
    benchmark_em = benchmark.get("student_exact_match")
    pr_b = benchmark.get("student_parse_fail_rate")
    pr_h = heldout.get("student_parse_fail_rate")
    avg_b = benchmark.get("student_avg_output_chars")
    avg_h = heldout.get("student_avg_output_chars")
    em_gap: float | None = None
    if isinstance(heldout_em, (int, float)) and isinstance(benchmark_em, (int, float)):
        em_gap = abs(float(heldout_em) - float(benchmark_em))

    parse_mean: float | None = None
    if isinstance(pr_b, (int, float)) and isinstance(pr_h, (int, float)):
        parse_mean = (float(pr_b) + float(pr_h)) / 2.0

    avg_chars_mean: float | None = None
    if isinstance(avg_b, (int, float)) and isinstance(avg_h, (int, float)):
        avg_chars_mean = (float(avg_b) + float(avg_h)) / 2.0

    return {
        "benchmark": benchmark,
        "heldout": heldout,
        "aggregate": {
            "heldout_em": float(heldout_em) if isinstance(heldout_em, (int, float)) else None,
            "benchmark_em": float(benchmark_em) if isinstance(benchmark_em, (int, float)) else None,
            "parse_fail_rate_mean": parse_mean,
            "avg_output_chars_mean": avg_chars_mean,
            "stability_em_gap": em_gap,
        },
    }


def _score_key(rec: dict[str, Any]) -> tuple[float, float, float, float, float]:
    agg = rec.get("evaluation", {}).get("aggregate", {}) if isinstance(rec.get("evaluation"), dict) else {}
    heldout_em = agg.get("heldout_em")
    benchmark_em = agg.get("benchmark_em")
    parse_mean = agg.get("parse_fail_rate_mean")
    avg_chars = agg.get("avg_output_chars_mean")
    gap = agg.get("stability_em_gap")

    # Primary: heldout EM desc. Secondary: benchmark EM desc.
    # Tie-breakers: lower parse fail, shorter outputs, smaller benchmark/heldout gap.
    return (
        float(heldout_em) if isinstance(heldout_em, (int, float)) else -1.0,
        float(benchmark_em) if isinstance(benchmark_em, (int, float)) else -1.0,
        -float(parse_mean) if isinstance(parse_mean, (int, float)) else -999.0,
        -float(avg_chars) if isinstance(avg_chars, (int, float)) else -999999.0,
        -float(gap) if isinstance(gap, (int, float)) else -999.0,
    )


def _select_winner(format_runs: list[dict[str, Any]]) -> dict[str, Any]:
    if not format_runs:
        raise RuntimeError("No format runs available for winner selection.")
    ordered = sorted(format_runs, key=_score_key, reverse=True)
    return ordered[0]


def _mix_rows(
    distill_rows: list[dict[str, Any]],
    gold_rows: list[dict[str, Any]],
    ratio: float,
    seed: int,
) -> tuple[list[dict[str, Any]], int, int]:
    if ratio <= 0.0 or not distill_rows:
        shuffled = list(distill_rows)
        rnd = random.Random(seed)
        rnd.shuffle(shuffled)
        return shuffled, len(distill_rows), 0

    cap_ratio = max(0.0, min(0.95, float(ratio)))
    if cap_ratio <= 0.0:
        shuffled = list(distill_rows)
        rnd = random.Random(seed)
        rnd.shuffle(shuffled)
        return shuffled, len(distill_rows), 0

    want_gold = int(round((cap_ratio / max(1e-6, 1.0 - cap_ratio)) * len(distill_rows)))
    want_gold = max(1, want_gold)
    if not gold_rows:
        return list(distill_rows), len(distill_rows), 0

    rnd = random.Random(seed)
    gold_pool = list(gold_rows)
    rnd.shuffle(gold_pool)
    selected_gold = gold_pool[: min(len(gold_pool), want_gold)]

    out = list(distill_rows) + selected_gold
    rnd.shuffle(out)
    return out, len(distill_rows), len(selected_gold)


def _format_metric(v: Any) -> str:
    if isinstance(v, (int, float)):
        return f"{float(v):.4f}"
    return "NA"


def _write_summary_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines: list[str] = []
    lines.append("# Distillation Phase Summary")
    lines.append("")
    lines.append(f"- Generated (UTC): `{payload.get('created_at_utc')}`")
    lines.append(f"- Phase: `{payload.get('phase_name')}`")
    lines.append(f"- Run ID: `{payload.get('pipeline_run_id')}`")
    lines.append(f"- Teacher checkpoint: `{payload.get('teacher_model')}`")
    lines.append(f"- Source comparison: `{payload.get('comparison_json')}`")
    lines.append("")

    filt = payload.get("filtered_subset_summary", {})
    lines.append("## Filter Summary")
    lines.append("")
    lines.append(f"- Total candidate rows: `{filt.get('total_rows')}`")
    lines.append(f"- Filter kept (teacher correct / student wrong): `{filt.get('filtered_rows')}`")
    lines.append(f"- Bucket counts before filtering: `{json.dumps(filt.get('bucket_counts_before', {}), ensure_ascii=False)}`")
    lines.append("")

    lines.append("## Dataset Variants")
    lines.append("")
    for item in payload.get("dataset_variants", []):
        lines.append(f"- `{item.get('target_format')}`: `{item.get('path')}` (rows={item.get('rows')})")
    lines.append("")

    lines.append("## Format Eval Table")
    lines.append("")
    lines.append("| Format | Held-out EM | Benchmark EM | ParseFail(mean) | AvgChars(mean) | EM Gap |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for rec in payload.get("format_eval_summary_table", []):
        lines.append(
            "| "
            + f"{rec.get('target_format')}"
            + " | "
            + f"{_format_metric(rec.get('heldout_em'))}"
            + " | "
            + f"{_format_metric(rec.get('benchmark_em'))}"
            + " | "
            + f"{_format_metric(rec.get('parse_fail_rate_mean'))}"
            + " | "
            + f"{_format_metric(rec.get('avg_output_chars_mean'))}"
            + " | "
            + f"{_format_metric(rec.get('stability_em_gap'))}"
            + " |"
        )
    lines.append("")

    winner = payload.get("winner_selection", {})
    lines.append("## Winner")
    lines.append("")
    lines.append(f"- Winner format: `{winner.get('winner_format')}`")
    lines.append(f"- Note: {winner.get('selection_note')}")
    lines.append("")

    lines.append("## Pure Vs Mixed")
    lines.append("")
    lines.append("| Mode | Held-out EM | Benchmark EM | ParseFail(mean) | AvgChars(mean) |")
    lines.append("|---|---:|---:|---:|---:|")
    for row in payload.get("pure_vs_mixed", {}).get("comparison_table", []):
        lines.append(
            "| "
            + f"{row.get('mode')}"
            + " | "
            + f"{_format_metric(row.get('heldout_em'))}"
            + " | "
            + f"{_format_metric(row.get('benchmark_em'))}"
            + " | "
            + f"{_format_metric(row.get('parse_fail_rate_mean'))}"
            + " | "
            + f"{_format_metric(row.get('avg_output_chars_mean'))}"
            + " |"
        )
    lines.append("")
    answers = payload.get("answers", {})
    lines.append(f"- Q1 (best format): `{answers.get('best_transfer_format')}`")
    lines.append(f"- Q2 (mixed beats pure): `{answers.get('mixed_beats_pure')}`")

    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    configure_logging()
    args = parse_args()

    teacher_model, comparison_json, run_id, experiment, parent_eval = _resolve_teacher_and_comparison(args)
    phase_slug = slugify(f"{args.phase_name}-{run_id}")

    out_root = Path(args.output_root)
    out_root.mkdir(parents=True, exist_ok=True)

    comparison_payload = _read_json(Path(comparison_json))
    rows = comparison_payload.get("sample_comparison", {}).get("examples", [])
    if not isinstance(rows, list):
        raise RuntimeError(f"Invalid sample_comparison/examples in {comparison_json}")

    bucket_before = _compute_bucket_counts(rows)
    filtered_rows = _filter_teacher_correct_student_wrong(rows, max_rows=int(args.max_filtered_rows))
    if not filtered_rows:
        raise RuntimeError("Filtered subset is empty (teacher correct / student wrong).")

    filtered_summary = {
        "total_rows": len(rows),
        "filtered_rows": len(filtered_rows),
        "bucket_counts_before": bucket_before,
        "excluded_counts": {
            "both correct": bucket_before.get("both correct", 0),
            "both wrong": bucket_before.get("both wrong", 0),
            "teacher wrong / student correct": bucket_before.get("teacher wrong / student correct", 0),
        },
    }

    variants: list[dict[str, Any]] = []
    variant_rows_map: dict[str, list[dict[str, Any]]] = {}
    variant_ids_map: dict[str, list[int]] = {}
    for fmt in ("answer_only", "short_rationale", "full_rationale"):
        rows_fmt, ids_fmt = _render_rows(
            filtered_rows,
            target_format=fmt,
            target_source="teacher",
            short_max_sentences=int(args.short_rationale_max_sentences),
            short_max_chars=int(args.short_rationale_max_chars),
            max_answer_chars=int(args.max_answer_chars),
        )
        if not rows_fmt:
            raise RuntimeError(f"No rows remained after rendering format={fmt}.")
        path = out_root / f"{phase_slug}_{fmt}.jsonl"
        _write_jsonl(path, rows_fmt)
        variant_rows_map[fmt] = rows_fmt
        variant_ids_map[fmt] = ids_fmt
        variants.append({"target_format": fmt, "path": str(path), "rows": len(rows_fmt)})

    # Ensure identical membership across 3 datasets.
    id_sets = []
    for fmt in ("answer_only", "short_rationale", "full_rationale"):
        id_sets.append(variant_ids_map[fmt])
    if not (id_sets[0] == id_sets[1] == id_sets[2]):
        raise RuntimeError("Dataset variants do not share identical membership/order.")

    shared_train_config = {
        "student_base_model": args.student_base_model,
        "student_epochs": args.student_epochs,
        "student_learning_rate": args.student_lr,
        "student_batch_size": args.student_batch_size,
        "student_grad_accum": args.student_grad_accum,
        "student_max_seq_length": args.student_max_seq_length,
        "student_max_train_samples": args.student_max_train_samples,
        "student_max_eval_samples": args.student_max_eval_samples,
        "student_seed": args.student_seed,
        "train_val_ratio": args.train_val_ratio,
        "split_seed": args.split_seed,
    }

    format_runs: list[dict[str, Any]] = []
    for fmt in ("answer_only", "short_rationale", "full_rationale"):
        train_jsonl = Path(next(v["path"] for v in variants if v["target_format"] == fmt))
        run_tag = f"format-{fmt}"
        train_info = _train_student(args, train_jsonl, run_tag=run_tag, phase_slug=phase_slug, dry_run=args.dry_run)
        eval_info = _evaluate_two_sets(
            args,
            teacher_model=teacher_model,
            student_model=str(train_info["merged_dir"]),
            run_label=run_tag,
            phase_slug=phase_slug,
            parent_eval=parent_eval,
            experiment=experiment,
            run_id=run_id,
            dry_run=args.dry_run,
        )
        format_runs.append(
            {
                "target_format": fmt,
                "dataset_path": str(train_jsonl),
                "training": train_info,
                "evaluation": eval_info,
            }
        )

    winner = _select_winner(format_runs)
    winner_fmt = str(winner.get("target_format"))

    # Step 7: pure distill vs mixed gold+distill for winner format.
    winner_distill_rows = variant_rows_map[winner_fmt]
    winner_gold_rows, _ = _render_rows(
        filtered_rows,
        target_format=winner_fmt,
        target_source="gold",
        short_max_sentences=int(args.short_rationale_max_sentences),
        short_max_chars=int(args.short_rationale_max_chars),
        max_answer_chars=int(args.max_answer_chars),
    )
    if not winner_gold_rows:
        raise RuntimeError("No gold rows available for winner pure-vs-mixed step.")

    winner_gold_path = out_root / f"{phase_slug}_{winner_fmt}_gold.jsonl"
    _write_jsonl(winner_gold_path, winner_gold_rows)

    mixed_rows, pure_count, mixed_gold_count = _mix_rows(
        winner_distill_rows,
        winner_gold_rows,
        ratio=float(args.mixed_gold_ratio),
        seed=int(args.mixed_seed),
    )
    mixed_path = out_root / f"{phase_slug}_{winner_fmt}_mixed.jsonl"
    _write_jsonl(mixed_path, mixed_rows)

    pure_path = out_root / f"{phase_slug}_{winner_fmt}_pure.jsonl"
    _write_jsonl(pure_path, winner_distill_rows)

    pure_train = _train_student(args, pure_path, run_tag=f"winner-{winner_fmt}-pure", phase_slug=phase_slug, dry_run=args.dry_run)
    pure_eval = _evaluate_two_sets(
        args,
        teacher_model=teacher_model,
        student_model=str(pure_train["merged_dir"]),
        run_label=f"winner-{winner_fmt}-pure",
        phase_slug=phase_slug,
        parent_eval=parent_eval,
        experiment=experiment,
        run_id=run_id,
        dry_run=args.dry_run,
    )

    mixed_train = _train_student(args, mixed_path, run_tag=f"winner-{winner_fmt}-mixed", phase_slug=phase_slug, dry_run=args.dry_run)
    mixed_eval = _evaluate_two_sets(
        args,
        teacher_model=teacher_model,
        student_model=str(mixed_train["merged_dir"]),
        run_label=f"winner-{winner_fmt}-mixed",
        phase_slug=phase_slug,
        parent_eval=parent_eval,
        experiment=experiment,
        run_id=run_id,
        dry_run=args.dry_run,
    )

    pure_held = pure_eval.get("aggregate", {}).get("heldout_em")
    mixed_held = mixed_eval.get("aggregate", {}).get("heldout_em")
    mixed_beats_pure = (
        isinstance(mixed_held, (int, float))
        and isinstance(pure_held, (int, float))
        and float(mixed_held) > float(pure_held)
    )

    format_eval_table = []
    for rec in format_runs:
        agg = rec.get("evaluation", {}).get("aggregate", {})
        format_eval_table.append(
            {
                "target_format": rec.get("target_format"),
                "heldout_em": agg.get("heldout_em"),
                "benchmark_em": agg.get("benchmark_em"),
                "parse_fail_rate_mean": agg.get("parse_fail_rate_mean"),
                "avg_output_chars_mean": agg.get("avg_output_chars_mean"),
                "stability_em_gap": agg.get("stability_em_gap"),
            }
        )

    pure_vs_mixed_table = [
        {
            "mode": "pure_distill",
            "heldout_em": pure_eval.get("aggregate", {}).get("heldout_em"),
            "benchmark_em": pure_eval.get("aggregate", {}).get("benchmark_em"),
            "parse_fail_rate_mean": pure_eval.get("aggregate", {}).get("parse_fail_rate_mean"),
            "avg_output_chars_mean": pure_eval.get("aggregate", {}).get("avg_output_chars_mean"),
        },
        {
            "mode": "mixed_gold_plus_distill",
            "heldout_em": mixed_eval.get("aggregate", {}).get("heldout_em"),
            "benchmark_em": mixed_eval.get("aggregate", {}).get("benchmark_em"),
            "parse_fail_rate_mean": mixed_eval.get("aggregate", {}).get("parse_fail_rate_mean"),
            "avg_output_chars_mean": mixed_eval.get("aggregate", {}).get("avg_output_chars_mean"),
        },
    ]

    winner_note = (
        "Selected by held-out EM (primary), benchmark EM (secondary), then lower parse-failure rate, "
        "shorter outputs, and smaller held-out/benchmark EM gap."
    )

    phase_summary = {
        "created_at_utc": utc_now(),
        "phase_name": args.phase_name,
        "pipeline_run_id": run_id,
        "experiment_name": experiment,
        "teacher_model": teacher_model,
        "comparison_json": comparison_json,
        "parent_eval": parent_eval,
        "filtered_subset_summary": filtered_summary,
        "dataset_variants": variants,
        "shared_train_config": shared_train_config,
        "format_runs": format_runs,
        "format_eval_summary_table": format_eval_table,
        "winner_selection": {
            "winner_format": winner_fmt,
            "selection_note": winner_note,
            "winner_metrics": winner.get("evaluation", {}).get("aggregate", {}),
        },
        "pure_vs_mixed": {
            "winner_format": winner_fmt,
            "pure_dataset_path": str(pure_path),
            "gold_dataset_path": str(winner_gold_path),
            "mixed_dataset_path": str(mixed_path),
            "mixed_gold_ratio": float(args.mixed_gold_ratio),
            "mixed_counts": {
                "pure_rows": pure_count,
                "mixed_gold_rows": mixed_gold_count,
                "mixed_total_rows": len(mixed_rows),
            },
            "pure_run": {"training": pure_train, "evaluation": pure_eval},
            "mixed_run": {"training": mixed_train, "evaluation": mixed_eval},
            "comparison_table": pure_vs_mixed_table,
        },
        "answers": {
            "best_transfer_format": winner_fmt,
            "mixed_beats_pure": mixed_beats_pure,
        },
    }

    summary_json_path = out_root / f"{phase_slug}_phase_summary.json"
    summary_md_path = out_root / f"{phase_slug}_phase_summary.md"
    summary_json_path.write_text(json.dumps(phase_summary, indent=2), encoding="utf-8")
    _write_summary_markdown(summary_md_path, phase_summary)

    filtered_summary_path = out_root / f"{phase_slug}_filtered_subset_summary.json"
    filtered_summary_path.write_text(json.dumps(filtered_summary, indent=2), encoding="utf-8")

    LOGGER.info("DONE phase summary: %s", summary_json_path)
    LOGGER.info("DONE phase report: %s", summary_md_path)
    LOGGER.info("DONE filtered summary: %s", filtered_summary_path)


if __name__ == "__main__":
    main()
