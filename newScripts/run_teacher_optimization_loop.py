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
from datetime import datetime, timezone
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_BASE_MODEL = "/media/prasanna/716F26140AED9B67/models/models--Qwen--Qwen2-Math-1.5B-Instruct"
DEFAULT_STUDENT_MODEL = "/media/prasanna/716F26140AED9B67/models/models--Qwen--Qwen2-0.5B-Instruct"
DEFAULT_DATASET_PATH = "/media/prasanna/716F26140AED9B67/datasets/GSM8K"

LOGGER = logging.getLogger("teacher_opt_loop")
SUBPROCESS_ECHO = False
COMMAND_TAIL_LINES = 25
HEARTBEAT_SECONDS = 30


def log_state(state: str, **fields: Any) -> None:
    payload = " ".join(f"{k}={fields[k]}" for k in sorted(fields))
    if payload:
        LOGGER.info("STATE | %s | %s", state, payload)
    else:
        LOGGER.info("STATE | %s", state)


@dataclass
class Candidate:
    cycle: int
    name: str
    epochs: float
    learning_rate: float
    max_seq_length: int
    train_jsonl: str | None = None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run coarse-to-fine teacher optimization with plateau gating.")
    p.add_argument("--dataset-path", type=str, default=DEFAULT_DATASET_PATH)
    p.add_argument("--split", choices=["train", "test"], default="test")
    p.add_argument("--max-samples", type=int, default=0)
    p.add_argument("--base-model", type=str, default=DEFAULT_BASE_MODEL)
    p.add_argument("--student-model", type=str, default=DEFAULT_STUDENT_MODEL)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--batch-size", type=str, default="1")
    p.add_argument("--num-fewshot", type=int, default=8)
    p.add_argument("--gen-max-toks", type=int, default=256)
    p.add_argument("--limit", type=str, default=None)

    p.add_argument("--work-root", type=str, default="output/teacher-opt")
    p.add_argument("--summary-out", type=str, default="newoutput/teacher_optimization_summary.json")
    p.add_argument("--max-cycles", type=int, default=2)
    p.add_argument("--hard-slice-rows", type=int, default=1024)
    p.add_argument("--experiment-name", type=str, default="teacher-opt")
    p.add_argument("--pipeline-run-id", type=str, default=None)
    p.add_argument(
        "--baseline-comparison-json",
        type=str,
        default=None,
        help="Path to before-distill comparison.json. If omitted, derived from experiment + run-id.",
    )

    p.add_argument("--teacher-em-target", type=float, default=0.20)
    p.add_argument("--plateau-delta", type=float, default=0.003)
    p.add_argument("--plateau-cycles", type=int, default=2)
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def slugify(name: str) -> str:
    s = name.strip().lower()
    out = []
    for ch in s:
        if ch.isalnum() or ch in ("-", "_", "."):
            out.append(ch)
        else:
            out.append("-")
    s2 = "".join(out).strip("-")
    while "--" in s2:
        s2 = s2.replace("--", "-")
    return s2 or "run"


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
        force=True,
    )


def run_cmd(cmd: list[str], dry_run: bool, retries: int = 0) -> None:
    cmd_text = shlex.join(cmd)
    LOGGER.info("CMD: %s", cmd_text)
    if dry_run:
        LOGGER.info("Dry run mode: command skipped.")
        return
    last_exc: subprocess.CalledProcessError | None = None
    for attempt in range(retries + 1):
        start = time.perf_counter()
        tail: deque[str] = deque(maxlen=max(10, COMMAND_TAIL_LINES))
        try:
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
                    if SUBPROCESS_ECHO:
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
            dt = time.perf_counter() - start
            LOGGER.info("Command succeeded in %.2fs", dt)
            return
        except subprocess.CalledProcessError as exc:
            last_exc = exc
            dt = time.perf_counter() - start
            LOGGER.error(
                "Command failed (attempt %d/%d) rc=%s after %.2fs",
                attempt + 1,
                retries + 1,
                exc.returncode,
                dt,
            )
            LOGGER.error("Failed command: %s", cmd_text)
            if tail:
                LOGGER.error("Last %d output lines:\n%s", len(tail), "\n".join(tail))
            if attempt < retries:
                LOGGER.warning("Retrying command...")
                continue
    assert last_exc is not None
    raise last_exc


def _latest_baseline_for_experiment(experiment_name: str) -> Path | None:
    root = Path("newoutput") / "lm_eval"
    if not root.exists():
        return None
    prefix = f"before-distill-{slugify(experiment_name)}-"
    candidates = sorted(
        [p / "comparison.json" for p in root.iterdir() if p.is_dir() and p.name.startswith(prefix)],
        key=lambda p: p.stat().st_mtime if p.exists() else 0.0,
        reverse=True,
    )
    for c in candidates:
        if c.exists():
            return c
    return None


def _run_id_from_baseline_eval_name(eval_name: str, experiment_name: str) -> str | None:
    prefix = f"before-distill-{slugify(experiment_name)}-"
    if not eval_name.startswith(prefix):
        return None
    rid = eval_name[len(prefix) :].strip()
    return rid or None


def parse_comparison(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def teacher_em(path: Path) -> float:
    if not path.exists():
        return 0.0
    payload = parse_comparison(path)
    val = payload.get("teacher_exact_match")
    return float(val) if isinstance(val, (int, float)) else 0.0


def teacher_ok_map(path: Path) -> dict[int, bool]:
    payload = parse_comparison(path)
    rows = payload.get("sample_comparison", {}).get("examples", [])
    out: dict[int, bool] = {}
    if isinstance(rows, list):
        for r in rows:
            idx = r.get("idx")
            if isinstance(idx, int):
                out[idx] = bool(r.get("teacher_ok", False))
    return out


def diff_counts(baseline_cmp: Path, candidate_cmp: Path) -> tuple[int, int]:
    base = teacher_ok_map(baseline_cmp)
    cand = teacher_ok_map(candidate_cmp)
    keys = set(base) & set(cand)
    improved = sum(1 for k in keys if (not base[k]) and cand[k])
    regressed = sum(1 for k in keys if base[k] and (not cand[k]))
    return improved, regressed


def coarse_candidates() -> list[Candidate]:
    return [
        Candidate(cycle=1, name="c1_e1.0_lr1e-4_s2048", epochs=1.0, learning_rate=1e-4, max_seq_length=2048),
        Candidate(cycle=1, name="c1_e2.0_lr5e-5_s2048", epochs=2.0, learning_rate=5e-5, max_seq_length=2048),
        Candidate(cycle=1, name="c1_e1.5_lr7.5e-5_s2048", epochs=1.5, learning_rate=7.5e-5, max_seq_length=2048),
        Candidate(cycle=1, name="c1_e1.5_lr1e-4_s3072", epochs=1.5, learning_rate=1e-4, max_seq_length=3072),
    ]


def fine_candidates(cycle: int, top_two: list[dict[str, Any]]) -> list[Candidate]:
    out: list[Candidate] = []
    seen: set[str] = set()
    for rank, t in enumerate(top_two, start=1):
        base = t["config"]
        for lr_mult in (0.75, 1.0):
            for ep_mult in (0.75, 1.0):
                epochs = max(0.5, round(float(base["epochs"]) * ep_mult, 2))
                lr = float(base["learning_rate"]) * lr_mult
                name = f"c{cycle}_top{rank}_e{epochs}_lr{lr:.2e}_s{int(base['max_seq_length'])}"
                if name in seen:
                    continue
                seen.add(name)
                out.append(
                    Candidate(
                        cycle=cycle,
                        name=name,
                        epochs=epochs,
                        learning_rate=lr,
                        max_seq_length=int(base["max_seq_length"]),
                    )
                )
    return out


def refinement_candidate(cycle: int, best: dict[str, Any], hard_jsonl: Path) -> Candidate:
    cfg = best["config"]
    return Candidate(
        cycle=cycle,
        name=f"c{cycle}_refine_hard_e1.0_lr{float(cfg['learning_rate']) * 0.5:.2e}",
        epochs=1.0,
        learning_rate=float(cfg["learning_rate"]) * 0.5,
        max_seq_length=int(cfg["max_seq_length"]),
        train_jsonl=str(hard_jsonl),
    )


def run_candidate(
    args: argparse.Namespace,
    cand: Candidate,
    run_id: str,
    parent_eval: str,
    stage: str,
) -> dict[str, Any]:
    run_root = Path(args.work_root) / cand.name
    train_dir = run_root / "full_sft"
    merged_dir = run_root / "merged"
    LOGGER.info(
        "Running candidate train | cycle=%d name=%s epochs=%s lr=%s seq=%s",
        cand.cycle,
        cand.name,
        cand.epochs,
        cand.learning_rate,
        cand.max_seq_length,
    )
    sft_started = time.perf_counter()
    log_state("sft_start", cycle=cand.cycle, candidate=cand.name, stage=stage, run_id=run_id)
    LOGGER.info(
        "========== SFT START | cycle=%d | candidate=%s ==========",
        cand.cycle,
        cand.name,
    )
    train_cmd = [
        "uv",
        "run",
        "python",
        "newScripts/train_teacher_full_sft_gsm8k.py",
        "--base-model",
        args.base_model,
        "--dataset-path",
        args.dataset_path,
        "--split",
        "train",
        "--epochs",
        str(cand.epochs),
        "--learning-rate",
        str(cand.learning_rate),
        "--max-seq-length",
        str(cand.max_seq_length),
        "--answer-style",
        "cot_final_marker",
        "--output-dir",
        str(train_dir),
        "--merged-output-dir",
        str(merged_dir),
        "--merge-16bit",
        "1",
    ]
    if cand.train_jsonl:
        train_cmd.extend(["--train-jsonl", cand.train_jsonl])
        train_cmd.extend(["--eval-jsonl", cand.train_jsonl])
    run_cmd(train_cmd, args.dry_run, retries=1)
    sft_elapsed = time.perf_counter() - sft_started
    log_state(
        "sft_end",
        cycle=cand.cycle,
        candidate=cand.name,
        stage=stage,
        run_id=run_id,
        elapsed_s=f"{sft_elapsed:.2f}",
    )
    LOGGER.info(
        "========== SFT END   | cycle=%d | candidate=%s | elapsed=%.2fs ==========",
        cand.cycle,
        cand.name,
        sft_elapsed,
    )

    eval_name = f"{slugify(stage)}-{cand.name}-{run_id}"
    LOGGER.info("Running candidate eval | cycle=%d name=%s eval=%s", cand.cycle, cand.name, eval_name)
    eval_started = time.perf_counter()
    log_state("eval_start", cycle=cand.cycle, candidate=cand.name, stage=stage, eval_name=eval_name)
    LOGGER.info(
        "========== EVAL START | cycle=%d | candidate=%s | eval=%s ==========",
        cand.cycle,
        cand.name,
        eval_name,
    )
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
        str(merged_dir),
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
        stage,
        "--experiment",
        args.experiment_name,
        "--run-id",
        run_id,
        "--parent-eval",
        parent_eval,
        "--overwrite-output",
    ]
    if args.limit:
        eval_cmd.extend(["--limit", str(args.limit)])
    run_cmd(eval_cmd, args.dry_run, retries=1)
    eval_elapsed = time.perf_counter() - eval_started
    log_state(
        "eval_end",
        cycle=cand.cycle,
        candidate=cand.name,
        stage=stage,
        eval_name=eval_name,
        elapsed_s=f"{eval_elapsed:.2f}",
    )
    LOGGER.info(
        "========== EVAL END   | cycle=%d | candidate=%s | eval=%s | elapsed=%.2fs ==========",
        cand.cycle,
        cand.name,
        eval_name,
        eval_elapsed,
    )

    cmp_path = Path("newoutput") / "lm_eval" / eval_name / "comparison.json"
    em = teacher_em(cmp_path) if cmp_path.exists() else 0.0
    return {
        "cycle": cand.cycle,
        "name": cand.name,
        "config": {
            "epochs": cand.epochs,
            "learning_rate": cand.learning_rate,
            "max_seq_length": cand.max_seq_length,
            "train_jsonl": cand.train_jsonl,
            "train_mode": "full_sft",
        },
        "train_dir": str(train_dir),
        "lora_dir": str(train_dir),  # legacy key retained for compatibility
        "merged_dir": str(merged_dir),
        "eval_name": eval_name,
        "comparison_json": str(cmp_path),
        "teacher_exact_match": em,
    }


def main() -> None:
    args = parse_args()
    configure_logging()

    run_id = args.pipeline_run_id or ""
    log_state("loop_start", experiment=args.experiment_name, run_id=run_id or "auto", max_cycles=args.max_cycles)
    LOGGER.info("Starting teacher optimization loop | experiment=%s run_id=%s", args.experiment_name, run_id)
    summary_path = Path(args.summary_out)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    Path(args.work_root).mkdir(parents=True, exist_ok=True)

    baseline_eval = f"before-distill-{slugify(args.experiment_name)}-{run_id}"
    if args.baseline_comparison_json:
        baseline_cmp = Path(args.baseline_comparison_json)
    else:
        explicit = Path("newoutput") / "lm_eval" / baseline_eval / "comparison.json"
        if explicit.exists():
            baseline_cmp = explicit
        else:
            auto = _latest_baseline_for_experiment(args.experiment_name)
            baseline_cmp = auto if auto is not None else explicit
    if not baseline_cmp.exists():
        raise FileNotFoundError(
            "Baseline comparison.json not found. Run before-distill first.\n"
            "Suggested command:\n"
            f"  uv run python newScripts/run_before_distill_eval.py --experiment-name {args.experiment_name} "
            f"--pipeline-run-id {run_id}\n"
            f"Expected baseline file: {baseline_cmp}"
        )
    baseline_eval = baseline_cmp.parent.name
    inferred_run_id = _run_id_from_baseline_eval_name(baseline_eval, args.experiment_name)
    if not args.pipeline_run_id and inferred_run_id:
        run_id = inferred_run_id
    log_state("baseline_resolved", baseline=str(baseline_cmp), run_id=run_id, eval_name=baseline_eval)
    LOGGER.info("Using baseline comparison: %s", baseline_cmp)

    results: list[dict[str, Any]] = []
    per_cycle: list[dict[str, Any]] = []
    cycle_gains: list[float] = []
    best: dict[str, Any] | None = None
    plateau_streak = 0

    prev_cycle_top2: list[dict[str, Any]] = []
    for cycle in range(1, args.max_cycles + 1):
        log_state("cycle_start", cycle=cycle, max_cycles=args.max_cycles)
        LOGGER.info("Cycle %d/%d", cycle, args.max_cycles)
        if cycle == 1:
            candidates = coarse_candidates()
        elif cycle == 2:
            candidates = fine_candidates(cycle, prev_cycle_top2 if prev_cycle_top2 else results[:2])
        else:
            if best is None:
                log_state("cycle_skip_no_best", cycle=cycle)
                break
            hard_jsonl = Path(args.work_root) / f"hard_slice_cycle_{cycle}.jsonl"
            log_state("hard_slice_build_start", cycle=cycle, source_comparison=str(best["comparison_json"]))
            hard_cmd = [
                "uv",
                "run",
                "python",
                "newScripts/build_distill_gsm8k.py",
                "--comparison-json",
                str(best["comparison_json"]),
                "--output-jsonl",
                str(hard_jsonl),
                "--mode",
                "hard_failures",
                "--max-rows",
                str(args.hard_slice_rows),
            ]
            run_cmd(hard_cmd, args.dry_run, retries=0)
            log_state("hard_slice_build_end", cycle=cycle, output_jsonl=str(hard_jsonl))
            candidates = [refinement_candidate(cycle, best, hard_jsonl)]

        log_state("cycle_candidates_ready", cycle=cycle, count=len(candidates))
        LOGGER.info("Cycle %d candidates: %s", cycle, ", ".join(c.name for c in candidates))

        cycle_runs: list[dict[str, Any]] = []
        prev_best_em = float(best["teacher_exact_match"]) if best is not None else teacher_em(baseline_cmp)
        for cand in candidates:
            parent_eval = str(best["eval_name"]) if best is not None else baseline_eval
            stage = "teacher-refine" if cycle >= 3 else "teacher-sft"
            rec = run_candidate(args, cand, run_id=run_id, parent_eval=parent_eval, stage=stage)
            cycle_runs.append(rec)
            results.append(rec)
            LOGGER.info(
                "Candidate done | cycle=%d name=%s teacher_em=%.6f",
                cycle,
                rec["name"],
                float(rec["teacher_exact_match"]),
            )
            log_state(
                "candidate_done",
                cycle=cycle,
                candidate=rec["name"],
                teacher_em=f"{float(rec['teacher_exact_match']):.6f}",
                eval_name=rec["eval_name"],
            )
            if best is None or float(rec["teacher_exact_match"]) > float(best["teacher_exact_match"]):
                best = rec
                log_state(
                    "best_updated",
                    cycle=cycle,
                    candidate=best["name"],
                    teacher_em=f"{float(best['teacher_exact_match']):.6f}",
                )
                LOGGER.info("New best candidate: %s (EM=%.6f)", best["name"], float(best["teacher_exact_match"]))

        cycle_runs.sort(key=lambda r: float(r["teacher_exact_match"]), reverse=True)
        prev_cycle_top2 = cycle_runs[:2]
        cycle_best = cycle_runs[0] if cycle_runs else None
        cycle_best_em = float(cycle_best["teacher_exact_match"]) if cycle_best else prev_best_em
        gain = cycle_best_em - prev_best_em
        cycle_gains.append(gain)
        plateau_streak = plateau_streak + 1 if gain < args.plateau_delta else 0

        per_cycle.append(
            {
                "cycle": cycle,
                "runs": cycle_runs,
                "cycle_best_em": cycle_best_em,
                "gain_vs_prev_best": gain,
                "plateau_streak": plateau_streak,
            }
        )

        if best is not None and baseline_cmp.exists() and Path(str(best["comparison_json"])).exists():
            improved, regressed = diff_counts(baseline_cmp, Path(str(best["comparison_json"])))
        else:
            improved, regressed = 0, 0
        gate_ready = (
            best is not None
            and float(best["teacher_exact_match"]) >= args.teacher_em_target
            and plateau_streak >= args.plateau_cycles
            and improved > regressed
        )
        log_state(
            "cycle_gate_check",
            cycle=cycle,
            gate_ready=gate_ready,
            improved=improved,
            regressed=regressed,
            plateau_streak=plateau_streak,
            cycle_gain=f"{gain:.6f}",
        )
        if gate_ready:
            LOGGER.info("Distill gate reached. Stopping loop early at cycle %d.", cycle)
            break

    best_cmp = Path(str(best["comparison_json"])) if best is not None else None
    if baseline_cmp.exists() and best_cmp and best_cmp.exists():
        improved, regressed = diff_counts(baseline_cmp, best_cmp)
    else:
        improved, regressed = 0, 0

    best_em = float(best["teacher_exact_match"]) if best is not None else 0.0
    summary = {
        "experiment_name": args.experiment_name,
        "pipeline_run_id": run_id,
        "base_model": args.base_model,
        "student_model": args.student_model,
        "baseline_comparison_json": str(baseline_cmp),
        "best_run": best,
        "best_teacher_exact_match": best_em,
        "teacher_em_target": args.teacher_em_target,
        "plateau_delta": args.plateau_delta,
        "plateau_cycles_required": args.plateau_cycles,
        "cycle_gains": cycle_gains,
        "improved_vs_baseline": improved,
        "regressed_vs_baseline": regressed,
        "improvements_dominate": improved > regressed,
        "target_met": best_em >= args.teacher_em_target,
        "plateau_met": len(cycle_gains) >= args.plateau_cycles
        and all(g < args.plateau_delta for g in cycle_gains[-args.plateau_cycles :]),
        "distill_gate_passed": (
            best_em >= args.teacher_em_target
            and len(cycle_gains) >= args.plateau_cycles
            and all(g < args.plateau_delta for g in cycle_gains[-args.plateau_cycles :])
            and improved > regressed
        ),
        "cycles": per_cycle,
        "all_runs": results,
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    log_state(
        "loop_end",
        summary_path=str(summary_path),
        best_teacher_em=f"{best_em:.6f}",
        total_cycles=len(per_cycle),
        total_runs=len(results),
    )
    LOGGER.info("Optimization summary written: %s", summary_path)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        LOGGER.exception("Unhandled exception in teacher optimization loop.")
        raise
