#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


DEFAULT_TEACHER_MODEL = "/media/prasanna/716F26140AED9B67/models/models--Qwen--Qwen2-Math-1.5B-Instruct"
DEFAULT_STUDENT_MODEL = "/media/prasanna/716F26140AED9B67/models/models--Qwen--Qwen2-0.5B-Instruct"
DEFAULT_DATASET_PATH = "/media/prasanna/716F26140AED9B67/datasets/GSM8K"

LOGGER = logging.getLogger("before_distill_eval")


def slugify(name: str) -> str:
    out = []
    for ch in name.strip().lower():
        if ch.isalnum() or ch in ("-", "_", "."):
            out.append(ch)
        else:
            out.append("-")
    s = "".join(out).strip("-")
    while "--" in s:
        s = s.replace("--", "-")
    return s or "run"


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
        force=True,
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run before-distill baseline eval using frozen lm-eval protocol.")
    p.add_argument("--dataset-path", type=str, default=DEFAULT_DATASET_PATH)
    p.add_argument("--split", choices=["train", "test"], default="test")
    p.add_argument("--max-samples", type=int, default=0)
    p.add_argument("--teacher-model", type=str, default=DEFAULT_TEACHER_MODEL)
    p.add_argument("--student-model", type=str, default=DEFAULT_STUDENT_MODEL)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--batch-size", type=str, default="1")
    p.add_argument("--num-fewshot", type=int, default=8)
    p.add_argument("--gen-max-toks", type=int, default=256)
    p.add_argument("--limit", type=str, default=None)
    p.add_argument("--experiment-name", type=str, default="teacher-opt")
    p.add_argument("--pipeline-run-id", type=str, default=None)
    p.add_argument("--eval-name", type=str, default=None)
    p.add_argument("--overwrite-output", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def run_cmd(cmd: list[str], dry_run: bool) -> None:
    LOGGER.info("CMD: %s", shlex.join(cmd))
    if dry_run:
        LOGGER.info("Dry run mode: command skipped.")
        return
    subprocess.run(cmd, check=True)


def main() -> None:
    configure_logging()
    args = parse_args()

    run_id = args.pipeline_run_id or ""
    eval_name = args.eval_name or f"before-distill-{slugify(args.experiment_name)}-{run_id}"

    cmd = [
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
        args.teacher_model,
        "--student-model",
        args.student_model,
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
        "before-distill",
        "--experiment",
        args.experiment_name,
        "--run-id",
        run_id,
    ]
    if args.overwrite_output:
        cmd.append("--overwrite-output")
    if args.limit:
        cmd.extend(["--limit", str(args.limit)])

    run_cmd(cmd, args.dry_run)

    comparison = Path("newoutput") / "lm_eval" / eval_name / "comparison.json"
    result = {
        "experiment_name": args.experiment_name,
        "pipeline_run_id": run_id,
        "eval_name": eval_name,
        "comparison_json": str(comparison),
    }
    LOGGER.info("Baseline run completed: %s", comparison)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
