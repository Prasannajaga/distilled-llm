#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
import shlex
import subprocess
import sys
import threading
import time
from collections import deque
from pathlib import Path

LOGGER = logging.getLogger("student_distill")
TAIL_LINES = 25
HEARTBEAT_SECONDS = 30


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run student distillation from best gated teacher summary.")
    p.add_argument("--summary-json", type=str, required=True, help="Output of run_teacher_optimization_loop.py")
    p.add_argument("--dataset-path", type=str, default="/media/prasanna/716F26140AED9B67/datasets/GSM8K")
    p.add_argument("--split", choices=["train", "test"], default="test")
    p.add_argument("--max-samples", type=int, default=0)
    p.add_argument("--distill-rows", type=int, default=0, help="Rows kept for student distill (<=0 means all).")
    p.add_argument("--num-fewshot", type=int, default=8)
    p.add_argument("--gen-max-toks", type=int, default=256)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--batch-size", type=str, default="1")
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


def main() -> None:
    configure_logging()
    args = parse_args()
    summary = json.loads(Path(args.summary_json).read_text(encoding="utf-8"))
    if not bool(summary.get("distill_gate_passed", False)):
        raise RuntimeError("Distill gate not passed. Run teacher optimization until milestone is reached.")

    best = summary.get("best_run", {})
    run_id = str(summary.get("pipeline_run_id", "")).strip() or "manual"
    experiment = str(summary.get("experiment_name", "teacher-opt")).strip() or "teacher-opt"
    parent_eval = str(best.get("eval_name", "")).strip() or None
    teacher_model = str(best.get("merged_dir", "")).strip()
    comparison_json = str(best.get("comparison_json", "")).strip()
    if not teacher_model or not comparison_json:
        raise RuntimeError("Summary missing best teacher merged_dir/comparison_json.")

    out_root = Path("newoutput") / "distill"
    out_root.mkdir(parents=True, exist_ok=True)
    distill_jsonl = out_root / "student_distill_data.jsonl"

    build_cmd = [
        "uv",
        "run",
        "python",
        "newScripts/build_distill_gsm8k.py",
        "--comparison-json",
        comparison_json,
        "--output-jsonl",
        str(distill_jsonl),
        "--mode",
        "distill_high_conf",
        "--max-rows",
        str(args.distill_rows),
    ]
    run_cmd(build_cmd, args.dry_run)

    student_lora = Path("output") / "student-gsm8k-distill-lora"
    student_merged = Path("output") / "student-gsm8k-distill-merged"
    train_cmd = [
        "uv",
        "run",
        "python",
        "newScripts/train_student_unsloth_distill.py",
        "--train-jsonl",
        str(distill_jsonl),
        "--output-dir",
        str(student_lora),
        "--merged-output-dir",
        str(student_merged),
        "--merge-16bit",
        "1",
    ]
    run_cmd(train_cmd, args.dry_run)

    eval_name = f"after-distill-{run_id}"
    eval_cmd = [
        "uv",
        "run",
        "python",
        "newScripts/run_lm_eval_custom.py",
        "--dataset-path",
        args.dataset_path,
        "--split",
        args.split,
        "--max-samples",
        str(args.max_samples),
        "--teacher-model",
        teacher_model,
        "--student-model",
        str(student_merged),
        "--device",
        args.device,
        "--batch-size",
        args.batch_size,
        "--num-fewshot",
        str(args.num_fewshot),
        "--gen-max-toks",
        str(args.gen_max_toks),
        "--fewshot-cot",
        "1",
        "--eval-name",
        eval_name,
        "--stage",
        "after-distill",
        "--experiment",
        experiment,
        "--run-id",
        run_id,
        "--parent-eval",
        parent_eval,
        "--overwrite-output",
    ]
    run_cmd(eval_cmd, args.dry_run)
    LOGGER.info("DONE distillation eval at newoutput/lm_eval/%s/comparison.json", eval_name)


if __name__ == "__main__":
    main()
