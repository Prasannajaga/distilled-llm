#!/usr/bin/env python3
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

from newScripts.common import (
    EXAMPLES_KEY,
    SAMPLE_COMPARISON_KEY,
    configure_logging,
    log_state,
    read_json,
    run_logged_command,
    slugify,
)

LOGGER = logging.getLogger("distill_eval")


def parse_student_behavior_metrics(comparison_payload: dict[str, Any]) -> dict[str, Any]:
    sample_comparison = comparison_payload.get(SAMPLE_COMPARISON_KEY, {})
    rows = sample_comparison.get(EXAMPLES_KEY, []) if isinstance(sample_comparison, dict) else []
    if not isinstance(rows, list):
        rows = []
    total_rows = len(rows)
    if total_rows <= 0:
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
        "loaded_rows": total_rows,
        "student_parse_fail_count": parse_fails,
        "student_parse_fail_rate": (parse_fails / total_rows) if total_rows > 0 else None,
        "student_avg_output_chars": (total_chars / total_rows) if total_rows > 0 else None,
    }


def eval_one_set(
    args: argparse.Namespace,
    *,
    teacher_model: str,
    student_model: str,
    eval_name: str,
    stage: str,
    dataset_path: str,
    dataset_jsonl: str | None,
    split: str,
    max_samples: int,
    parent_eval: str | None,
    experiment: str,
    run_id: str,
    dry_run: bool,
    overlap_references: list[str] | None = None,
) -> dict[str, Any]:
    cmd = [
        "uv",
        "run",
        "python",
        "-m",
        "newScripts.run_lm_eval_custom",
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
    if dataset_jsonl:
        cmd.extend(["--dataset-jsonl", str(dataset_jsonl)])
    if parent_eval:
        cmd.extend(["--parent-eval", parent_eval])
    if args.limit:
        cmd.extend(["--limit", str(args.limit)])
    if bool(getattr(args, "allow_env_limit", False)):
        cmd.append("--allow-env-limit")
    if bool(getattr(args, "allow_partial_eval", False)):
        cmd.append("--allow-partial-eval")
    if hasattr(args, "overlap_threshold"):
        cmd.extend(["--overlap-threshold", str(args.overlap_threshold)])
    if bool(getattr(args, "allow_overlap", False)):
        cmd.append("--allow-overlap")
    for ref in overlap_references or []:
        cmd.extend(["--overlap-reference", str(ref)])

    log_state(
        LOGGER,
        "set_eval_start",
        eval_name=eval_name,
        stage=stage,
        split=split,
        max_samples=max_samples,
    )
    run_logged_command(cmd, logger=LOGGER, dry_run=dry_run)

    comparison_path = Path("newoutput") / "lm_eval" / eval_name / "comparison.json"
    if dry_run:
        return {
            "eval_name": eval_name,
            "stage": stage,
            "dataset_path": dataset_path,
            "dataset_jsonl": dataset_jsonl,
            "split": split,
            "max_samples": max_samples,
            "comparison_json": str(comparison_path),
            "student_exact_match": None,
            "loaded_rows": 0,
            "student_parse_fail_count": 0,
            "student_parse_fail_rate": None,
            "student_avg_output_chars": None,
        }

    payload = read_json(comparison_path)
    behavior = parse_student_behavior_metrics(payload)
    exact_match_raw = payload.get("student_exact_match")
    exact_match = float(exact_match_raw) if isinstance(exact_match_raw, (int, float)) else None
    output = {
        "eval_name": eval_name,
        "stage": stage,
        "dataset_path": dataset_path,
        "dataset_jsonl": dataset_jsonl,
        "split": split,
        "max_samples": max_samples,
        "comparison_json": str(comparison_path),
        "student_exact_match": exact_match,
    }
    output.update(behavior)
    log_state(
        LOGGER,
        "set_eval_end",
        eval_name=eval_name,
        exact_match=exact_match if exact_match is not None else "NA",
        loaded_rows=output.get("loaded_rows", 0),
    )
    return output


def aggregate_eval_results(
    benchmark: dict[str, Any],
    heldout: dict[str, Any],
    final_eval: dict[str, Any] | None = None,
) -> dict[str, Any]:
    heldout_em = heldout.get("student_exact_match")
    benchmark_em = benchmark.get("student_exact_match")
    final_em = final_eval.get("student_exact_match") if isinstance(final_eval, dict) else None
    benchmark_parse = benchmark.get("student_parse_fail_rate")
    heldout_parse = heldout.get("student_parse_fail_rate")
    final_parse = final_eval.get("student_parse_fail_rate") if isinstance(final_eval, dict) else None
    benchmark_chars = benchmark.get("student_avg_output_chars")
    heldout_chars = heldout.get("student_avg_output_chars")
    final_chars = final_eval.get("student_avg_output_chars") if isinstance(final_eval, dict) else None

    stability_em_gap: float | None = None
    if isinstance(heldout_em, (int, float)) and isinstance(benchmark_em, (int, float)):
        stability_em_gap = abs(float(heldout_em) - float(benchmark_em))

    parse_fail_rate_mean: float | None = None
    parse_values = [v for v in (benchmark_parse, heldout_parse, final_parse) if isinstance(v, (int, float))]
    if parse_values:
        parse_fail_rate_mean = sum(float(v) for v in parse_values) / float(len(parse_values))

    avg_output_chars_mean: float | None = None
    char_values = [v for v in (benchmark_chars, heldout_chars, final_chars) if isinstance(v, (int, float))]
    if char_values:
        avg_output_chars_mean = sum(float(v) for v in char_values) / float(len(char_values))

    return {
        "heldout_em": float(heldout_em) if isinstance(heldout_em, (int, float)) else None,
        "benchmark_em": float(benchmark_em) if isinstance(benchmark_em, (int, float)) else None,
        "final_em": float(final_em) if isinstance(final_em, (int, float)) else None,
        "parse_fail_rate_mean": parse_fail_rate_mean,
        "avg_output_chars_mean": avg_output_chars_mean,
        "stability_em_gap": stability_em_gap,
    }


def evaluate_benchmark_and_heldout(
    args: argparse.Namespace,
    *,
    teacher_model: str,
    student_model: str,
    run_label: str,
    phase_slug: str,
    parent_eval: str | None,
    experiment: str,
    run_id: str,
    dry_run: bool,
    overlap_references: list[str] | None = None,
) -> dict[str, Any]:
    run_slug = slugify(f"{phase_slug}-{run_label}")

    benchmark_eval_name = slugify(f"distill-benchmark-{run_slug}-{run_id}")
    benchmark = eval_one_set(
        args,
        teacher_model=teacher_model,
        student_model=student_model,
        eval_name=benchmark_eval_name,
        stage="distill-format-benchmark",
        dataset_path=args.benchmark_dataset_path,
        dataset_jsonl=(args.benchmark_dataset_jsonl if hasattr(args, "benchmark_dataset_jsonl") else None),
        split=args.benchmark_split,
        max_samples=int(args.benchmark_max_samples),
        parent_eval=parent_eval,
        experiment=experiment,
        run_id=run_id,
        dry_run=dry_run,
        overlap_references=overlap_references,
    )

    heldout_dataset_path = args.heldout_dataset_path or args.benchmark_dataset_path
    heldout_dataset_jsonl = getattr(args, "heldout_dataset_jsonl", None) or None
    heldout_eval_name = slugify(f"distill-heldout-{run_slug}-{run_id}")
    heldout = eval_one_set(
        args,
        teacher_model=teacher_model,
        student_model=student_model,
        eval_name=heldout_eval_name,
        stage="distill-format-heldout",
        dataset_path=heldout_dataset_path,
        dataset_jsonl=heldout_dataset_jsonl,
        split=args.heldout_split,
        max_samples=int(args.heldout_max_samples),
        parent_eval=parent_eval,
        experiment=experiment,
        run_id=run_id,
        dry_run=dry_run,
        overlap_references=overlap_references,
    )

    final_eval: dict[str, Any] | None = None
    final_dataset_jsonl = getattr(args, "final_eval_jsonl", None) or None
    if final_dataset_jsonl:
        final_eval_name = slugify(f"distill-final-{run_slug}-{run_id}")
        final_eval = eval_one_set(
            args,
            teacher_model=teacher_model,
            student_model=student_model,
            eval_name=final_eval_name,
            stage="distill-format-final",
            dataset_path=args.benchmark_dataset_path,
            dataset_jsonl=final_dataset_jsonl,
            split=args.benchmark_split,
            max_samples=int(getattr(args, "final_eval_max_samples", args.benchmark_max_samples)),
            parent_eval=parent_eval,
            experiment=experiment,
            run_id=run_id,
            dry_run=dry_run,
            overlap_references=overlap_references,
        )

    aggregate = aggregate_eval_results(benchmark, heldout, final_eval=final_eval)
    log_state(
        LOGGER,
        "benchmark_heldout_aggregate",
        run_label=run_label,
        benchmark_em=aggregate.get("benchmark_em", "NA"),
        heldout_em=aggregate.get("heldout_em", "NA"),
        final_em=aggregate.get("final_em", "NA"),
    )
    payload: dict[str, Any] = {
        "benchmark": benchmark,
        "heldout": heldout,
        "aggregate": aggregate,
    }
    if final_eval is not None:
        payload["final"] = final_eval
    return payload


def score_key(rec: dict[str, Any]) -> tuple[float, float, float, float, float, float]:
    evaluation = rec.get("evaluation", {}) if isinstance(rec.get("evaluation"), dict) else {}
    aggregate = evaluation.get("aggregate", {}) if isinstance(evaluation.get("aggregate"), dict) else {}
    final_em = aggregate.get("final_em")
    heldout_em = aggregate.get("heldout_em")
    benchmark_em = aggregate.get("benchmark_em")
    parse_mean = aggregate.get("parse_fail_rate_mean")
    avg_chars = aggregate.get("avg_output_chars_mean")
    gap = aggregate.get("stability_em_gap")
    return (
        float(final_em) if isinstance(final_em, (int, float)) else -1.0,
        float(heldout_em) if isinstance(heldout_em, (int, float)) else -1.0,
        float(benchmark_em) if isinstance(benchmark_em, (int, float)) else -1.0,
        -float(parse_mean) if isinstance(parse_mean, (int, float)) else -999.0,
        -float(avg_chars) if isinstance(avg_chars, (int, float)) else -999999.0,
        -float(gap) if isinstance(gap, (int, float)) else -999.0,
    )


def select_winner(format_runs: list[dict[str, Any]]) -> dict[str, Any]:
    if not format_runs:
        raise RuntimeError("No format runs available for winner selection.")
    return sorted(format_runs, key=score_key, reverse=True)[0]


if __name__ == "__main__":
    configure_logging("distill_eval")
    raise SystemExit("distill_eval.py is an import-only helper module.")
