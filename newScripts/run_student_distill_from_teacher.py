#!/usr/bin/env python3
from __future__ import annotations

import argparse
import logging
import random
import shlex
from pathlib import Path
from typing import Any

from datasets import Dataset, load_dataset
from newScripts.common import (
    ANSWER_FIELD,
    BUCKET_BOTH_CORRECT,
    BUCKET_TEACHER_CORRECT_STUDENT_WRONG,
    BUCKET_TEACHER_WRONG_STUDENT_CORRECT,
    FORMAT_ANSWER_ONLY,
    IDX_FIELD,
    QUESTION_FIELD,
    SAMPLE_COMPARISON_KEY,
    TARGET_FORMATS,
    TARGET_SOURCE_GOLD,
    TARGET_SOURCE_TEACHER,
    build_phase_paths,
    build_example_keys,
    compute_overlap_report,
    configure_logging,
    iter_jsonl,
    log_state,
    read_json,
    render_dataset_rows,
    run_logged_command,
    slugify,
    split_rows_train_val,
    utc_now,
    write_json,
    write_jsonl_rows,
)
from newScripts.distill_eval import evaluate_benchmark_and_heldout, select_winner

LOGGER = logging.getLogger("student_distill")
DEFAULT_DATASET_PATH = "/media/prasanna/716F26140AED9B67/datasets/GSM8K"
RECIPE_DEFINITIONS: dict[str, dict[str, float]] = {
    "recipe_pure_target": {
        BUCKET_TEACHER_CORRECT_STUDENT_WRONG: 1.00,
        BUCKET_BOTH_CORRECT: 0.00,
        BUCKET_TEACHER_WRONG_STUDENT_CORRECT: 0.00,
    },
    "recipe_mix_b": {
        BUCKET_TEACHER_CORRECT_STUDENT_WRONG: 0.70,
        BUCKET_BOTH_CORRECT: 0.20,
        BUCKET_TEACHER_WRONG_STUDENT_CORRECT: 0.10,
    },
    "recipe_mix_c": {
        BUCKET_TEACHER_CORRECT_STUDENT_WRONG: 0.60,
        BUCKET_BOTH_CORRECT: 0.30,
        BUCKET_TEACHER_WRONG_STUDENT_CORRECT: 0.10,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run controlled distillation phase from a dataset manifest: train format variants, "
            "evaluate benchmark/heldout, select winner, and compare pure vs mixed datasets."
        )
    )
    parser.add_argument("--dataset-manifest-json", type=str, required=True)
    parser.add_argument("--summary-json", type=str, default=None, help="Optional teacher summary JSON with best_run fields.")
    parser.add_argument("--teacher-model", type=str, default=None, help="Pinned teacher checkpoint/model path.")
    parser.add_argument("--experiment-name", type=str, default="distill-format-ablation")
    parser.add_argument("--pipeline-run-id", type=str, default="")
    parser.add_argument("--parent-eval", type=str, default=None)

    parser.add_argument("--student-base-model", type=str, default="")
    parser.add_argument("--student-epochs", type=float, default=0.75)
    parser.add_argument("--student-lr", type=float, default=8e-5)
    parser.add_argument("--student-batch-size", type=int, default=4)
    parser.add_argument("--student-grad-accum", type=int, default=4)
    parser.add_argument("--student-max-seq-length", type=int, default=2048)
    parser.add_argument("--student-packing", type=int, choices=[0, 1], default=0)
    parser.add_argument("--student-max-train-samples", type=int, default=0)
    parser.add_argument("--student-max-eval-samples", type=int, default=512)
    parser.add_argument("--student-seed", type=int, default=42)
    parser.add_argument("--train-val-ratio", type=float, default=0.1)
    parser.add_argument("--split-seed", type=int, default=42)

    parser.add_argument("--benchmark-dataset-path", type=str, default=DEFAULT_DATASET_PATH)
    parser.add_argument("--benchmark-split", choices=["train", "test"], default="test")
    parser.add_argument("--benchmark-max-samples", type=int, default=500)
    parser.add_argument("--benchmark-dataset-jsonl", type=str, default="")
    parser.add_argument("--heldout-dataset-path", type=str, default="")
    parser.add_argument("--heldout-split", choices=["train", "test"], default="train")
    parser.add_argument("--heldout-max-samples", type=int, default=500)
    parser.add_argument("--heldout-dataset-jsonl", type=str, default="")
    parser.add_argument("--final-eval-jsonl", type=str, default="")
    parser.add_argument("--final-eval-max-samples", type=int, default=500)
    parser.add_argument("--frozen-eval-size", type=int, default=500)
    parser.add_argument("--rebuild-frozen-eval", action="store_true")
    parser.add_argument(
        "--frozen-eval-source",
        choices=["benchmark", "heldout"],
        default="benchmark",
        help="Source split used to materialize frozen clean eval when --final-eval-jsonl is not provided.",
    )

    parser.add_argument("--num-fewshot", type=int, default=8)
    parser.add_argument("--gen-max-toks", type=int, default=256)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--batch-size", type=str, default="1")
    parser.add_argument("--limit", type=str, default=None)
    parser.add_argument("--allow-env-limit", action="store_true")
    parser.add_argument("--allow-partial-eval", action="store_true")
    parser.add_argument("--overlap-threshold", type=float, default=0.0)
    parser.add_argument("--allow-overlap", action="store_true")

    parser.add_argument("--recipe-seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _search_arrow(root: Path, split: str) -> Path | None:
    target_name = f"gsm8k-{split}.arrow"
    if not root.exists() or not root.is_dir():
        return None
    direct = root / target_name
    if direct.exists():
        return direct
    matches = sorted(root.rglob(target_name))
    return matches[0] if matches else None


def _resolve_split_arrow(dataset_path: str, split: str) -> Path:
    requested = Path(dataset_path)
    if requested.exists():
        if requested.is_file():
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
    raise FileNotFoundError(f"Could not resolve gsm8k-{split}.arrow from {dataset_path}")


def _load_dataset_rows(dataset_path: str, split: str, max_samples: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        split_arrow = _resolve_split_arrow(dataset_path, split)
        ds = Dataset.from_file(str(split_arrow))
        for row in ds:
            q = str(row.get(QUESTION_FIELD, "")).strip()
            a = str(row.get(ANSWER_FIELD, "")).strip()
            if not q or not a:
                continue
            rows.append({QUESTION_FIELD: q, ANSWER_FIELD: a})
            if max_samples > 0 and len(rows) >= max_samples:
                break
        return rows
    except FileNotFoundError:
        ds = load_dataset("openai/gsm8k", "main", split=split)
        for row in ds:
            q = str(row.get(QUESTION_FIELD, "")).strip()
            a = str(row.get(ANSWER_FIELD, "")).strip()
            if not q or not a:
                continue
            rows.append({QUESTION_FIELD: q, ANSWER_FIELD: a})
            if max_samples > 0 and len(rows) >= max_samples:
                break
        return rows


def _load_source_examples(dataset_manifest: dict[str, Any]) -> list[dict[str, Any]]:
    comparison_json = str(dataset_manifest.get("comparison_json") or "").strip()
    if not comparison_json:
        raise RuntimeError("Dataset manifest is missing comparison_json.")
    payload = read_json(comparison_json)
    sample_cmp = payload.get(SAMPLE_COMPARISON_KEY, {})
    rows = sample_cmp.get("examples", []) if isinstance(sample_cmp, dict) else []
    if not isinstance(rows, list):
        raise RuntimeError(f"Invalid sample comparison examples in {comparison_json}")
    return [dict(row) for row in rows if isinstance(row, dict)]


def _bucket_label(row: dict[str, Any]) -> str:
    teacher_ok = bool(row.get("teacher_ok", False))
    student_ok = bool(row.get("student_ok", False))
    if teacher_ok and not student_ok:
        return BUCKET_TEACHER_CORRECT_STUDENT_WRONG
    if (not teacher_ok) and student_ok:
        return BUCKET_TEACHER_WRONG_STUDENT_CORRECT
    if teacher_ok and student_ok:
        return BUCKET_BOTH_CORRECT
    return "both wrong"


def _build_frozen_eval_jsonl(
    *,
    output_jsonl: Path,
    source_rows: list[dict[str, Any]],
    candidate_rows: list[dict[str, Any]],
    max_rows: int,
) -> dict[str, Any]:
    source_keys = set(build_example_keys(source_rows, id_fields=()))
    frozen_rows: list[dict[str, Any]] = []
    kept_keys: list[str] = []
    for row in candidate_rows:
        key = build_example_keys([row], id_fields=())[0]
        if key in source_keys:
            continue
        frozen_rows.append({QUESTION_FIELD: str(row.get(QUESTION_FIELD, "")), ANSWER_FIELD: str(row.get(ANSWER_FIELD, ""))})
        kept_keys.append(key)
        if max_rows > 0 and len(frozen_rows) >= max_rows:
            break
    if not frozen_rows:
        raise RuntimeError("Frozen eval set is empty after overlap filtering.")
    write_jsonl_rows(output_jsonl, frozen_rows)
    return {
        "frozen_eval_jsonl": str(output_jsonl),
        "rows": len(frozen_rows),
        "excluded_due_to_overlap": max(0, len(candidate_rows) - len(frozen_rows)),
        "frozen_keys_preview": kept_keys[:20],
    }


def _materialize_train_val_split(
    *,
    input_jsonl: Path,
    output_root: Path,
    run_tag: str,
    val_ratio: float,
    seed: int,
) -> dict[str, Any]:
    rows = [dict(row) for row in iter_jsonl(input_jsonl)]
    train_rows, val_rows, train_idx, val_idx = split_rows_train_val(rows, val_ratio=val_ratio, seed=seed)
    run_slug = slugify(run_tag)
    train_path = output_root / f"{run_slug}.train.jsonl"
    val_path = output_root / f"{run_slug}.val.jsonl"
    write_jsonl_rows(train_path, train_rows)
    if val_rows:
        write_jsonl_rows(val_path, val_rows)
    else:
        val_path = train_path
    return {
        "input_jsonl": str(input_jsonl),
        "train_jsonl": str(train_path),
        "val_jsonl": str(val_path),
        "train_rows": len(train_rows),
        "val_rows": len(val_rows),
        "train_indices": train_idx,
        "val_indices": val_idx,
        "seed": int(seed),
        "val_ratio": float(val_ratio),
    }


def _rows_by_bucket(source_rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {
        BUCKET_TEACHER_CORRECT_STUDENT_WRONG: [],
        BUCKET_BOTH_CORRECT: [],
        BUCKET_TEACHER_WRONG_STUDENT_CORRECT: [],
    }
    for row in source_rows:
        bucket = _bucket_label(row)
        if bucket in out:
            out[bucket].append(dict(row))
    return out


def _sample_bucket_rows(
    rows: list[dict[str, Any]],
    *,
    count: int,
    seed: int,
    salt: str,
) -> list[dict[str, Any]]:
    if count <= 0:
        return []
    if len(rows) < count:
        raise RuntimeError(f"Bucket '{salt}' has insufficient rows: requested={count} available={len(rows)}")
    rng = random.Random(f"{seed}:{salt}")
    indices = list(range(len(rows)))
    rng.shuffle(indices)
    chosen = sorted(indices[:count])
    return [dict(rows[i]) for i in chosen]


def _render_rows_for_bucket_training(
    *,
    rows: list[dict[str, Any]],
    target_format: str,
    target_source: str,
    dataset_manifest: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[int]]:
    render_config = dataset_manifest.get("render_config", {})
    if not isinstance(render_config, dict):
        render_config = {}
    return render_dataset_rows(
        rows,
        target_format=target_format,
        target_source=target_source,
        short_max_sentences=int(render_config.get("short_rationale_max_sentences", 3)),
        short_max_chars=int(render_config.get("short_rationale_max_chars", 360)),
        max_answer_chars=int(render_config.get("max_answer_chars", 2000)),
    )


def _build_recipe_datasets(
    *,
    phase_paths: Any,
    dataset_manifest: dict[str, Any],
    winner_format: str,
    seed: int,
) -> tuple[Path, dict[str, Any]]:
    source_rows = _load_source_examples(dataset_manifest)
    pools = _rows_by_bucket(source_rows)
    target_total = len(pools[BUCKET_TEACHER_CORRECT_STUDENT_WRONG])
    if target_total <= 0:
        raise RuntimeError("Target bucket is empty; cannot build recipe datasets.")

    recipe_root = phase_paths.datasets_root / "recipes" / winner_format
    recipe_root.mkdir(parents=True, exist_ok=True)
    recipe_records: dict[str, Any] = {}
    for recipe_name, ratios in RECIPE_DEFINITIONS.items():
        tc_sw_count = int(round(target_total * float(ratios.get(BUCKET_TEACHER_CORRECT_STUDENT_WRONG, 0.0))))
        bc_count = int(round(target_total * float(ratios.get(BUCKET_BOTH_CORRECT, 0.0))))
        tw_sc_count = max(0, target_total - tc_sw_count - bc_count)
        row_plan = {
            BUCKET_TEACHER_CORRECT_STUDENT_WRONG: tc_sw_count,
            BUCKET_BOTH_CORRECT: bc_count,
            BUCKET_TEACHER_WRONG_STUDENT_CORRECT: tw_sc_count,
        }

        sampled_rows: list[dict[str, Any]] = []
        sampled_row_ids: list[int] = []
        per_bucket_counts: dict[str, int] = {}
        for bucket_name in (
            BUCKET_TEACHER_CORRECT_STUDENT_WRONG,
            BUCKET_BOTH_CORRECT,
            BUCKET_TEACHER_WRONG_STUDENT_CORRECT,
        ):
            needed = int(row_plan[bucket_name])
            if needed <= 0:
                per_bucket_counts[bucket_name] = 0
                continue
            source_for_bucket = (
                TARGET_SOURCE_TEACHER
                if bucket_name == BUCKET_TEACHER_CORRECT_STUDENT_WRONG
                else TARGET_SOURCE_GOLD
            )
            sampled = _sample_bucket_rows(
                pools[bucket_name],
                count=needed,
                seed=seed,
                salt=f"{recipe_name}:{bucket_name}",
            )
            rendered_rows, rendered_ids = _render_rows_for_bucket_training(
                rows=sampled,
                target_format=winner_format,
                target_source=source_for_bucket,
                dataset_manifest=dataset_manifest,
            )
            if len(rendered_rows) != needed:
                raise RuntimeError(
                    f"Rendered row count mismatch for recipe={recipe_name} bucket={bucket_name}: "
                    f"needed={needed} rendered={len(rendered_rows)}"
                )
            sampled_rows.extend(rendered_rows)
            sampled_row_ids.extend(rendered_ids)
            per_bucket_counts[bucket_name] = len(rendered_rows)

        output_jsonl = recipe_root / f"{recipe_name}.jsonl"
        write_jsonl_rows(output_jsonl, sampled_rows)
        recipe_records[recipe_name] = {
            "dataset_path": str(output_jsonl),
            "rows": len(sampled_rows),
            "bucket_counts": per_bucket_counts,
            "target_ratios": ratios,
            "source_ids": sampled_row_ids,
        }

    manifest = {
        "created_at_utc": utc_now(),
        "winner_format": winner_format,
        "seed": int(seed),
        "target_total_rows": target_total,
        "recipes": recipe_records,
        "bucket_pool_sizes": {
            BUCKET_TEACHER_CORRECT_STUDENT_WRONG: len(pools[BUCKET_TEACHER_CORRECT_STUDENT_WRONG]),
            BUCKET_BOTH_CORRECT: len(pools[BUCKET_BOTH_CORRECT]),
            BUCKET_TEACHER_WRONG_STUDENT_CORRECT: len(pools[BUCKET_TEACHER_WRONG_STUDENT_CORRECT]),
        },
    }
    manifest_path = phase_paths.manifests_root / f"bucket_recipe_manifest_{winner_format}.json"
    write_json(manifest_path, manifest)
    return manifest_path, manifest


def _make_overlap_refs(train_jsonl: Path, val_jsonl: Path | None = None) -> list[str]:
    refs = [f"distill_train={train_jsonl}"]
    if val_jsonl is not None:
        refs.append(f"distill_val={val_jsonl}")
    return refs


def _load_eval_examples(comparison_json: str) -> list[dict[str, Any]]:
    payload = read_json(comparison_json)
    sample_cmp = payload.get(SAMPLE_COMPARISON_KEY, {})
    rows = sample_cmp.get("examples", []) if isinstance(sample_cmp, dict) else []
    if not isinstance(rows, list):
        return []
    return [dict(row) for row in rows if isinstance(row, dict)]


def _question_key(row: dict[str, Any]) -> str:
    return str(row.get(QUESTION_FIELD, "")).strip().lower()


def _compute_recovery_damage(
    baseline_rows: list[dict[str, Any]],
    candidate_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    baseline_by_q = {_question_key(row): row for row in baseline_rows}
    candidate_by_q = {_question_key(row): row for row in candidate_rows}
    shared_keys = [k for k in baseline_by_q.keys() if k in candidate_by_q]
    recovery = 0
    damage_both_correct = 0
    damage_teacher_wrong_student_correct = 0
    for key in shared_keys:
        base = baseline_by_q[key]
        curr = candidate_by_q[key]
        teacher_ok = bool(base.get("teacher_ok", False))
        base_student_ok = bool(base.get("student_ok", False))
        curr_student_ok = bool(curr.get("student_ok", False))
        if teacher_ok and (not base_student_ok) and curr_student_ok:
            recovery += 1
        if teacher_ok and base_student_ok and (not curr_student_ok):
            damage_both_correct += 1
        if (not teacher_ok) and base_student_ok and (not curr_student_ok):
            damage_teacher_wrong_student_correct += 1
    return {
        "shared_rows": len(shared_keys),
        "recovery_count_teacher_correct_student_wrong": recovery,
        "damage_count_both_correct": damage_both_correct,
        "damage_count_teacher_wrong_student_correct": damage_teacher_wrong_student_correct,
    }


def _resolve_teacher_and_context(
    args: argparse.Namespace,
    dataset_manifest: dict[str, Any],
) -> tuple[str, str, str, str | None]:
    summary: dict[str, Any] = {}
    if args.summary_json:
        summary = read_json(args.summary_json)
    best = summary.get("best_run", {}) if isinstance(summary.get("best_run"), dict) else {}

    teacher_model = str(args.teacher_model or best.get("merged_dir") or "").strip()
    if not teacher_model:
        raise RuntimeError("Teacher model is required. Pass --teacher-model or --summary-json with best_run.merged_dir.")

    run_id = str(args.pipeline_run_id or dataset_manifest.get("pipeline_run_id") or summary.get("pipeline_run_id") or "").strip()
    if not run_id:
        run_id = utc_now().replace(":", "").replace("-", "")[:15]

    experiment = str(args.experiment_name or summary.get("experiment_name") or "distill-format-ablation").strip()
    parent_eval = str(args.parent_eval or best.get("eval_name") or "").strip() or None
    return teacher_model, run_id, experiment, parent_eval


def _phase_paths_from_manifest(dataset_manifest: dict[str, Any]) -> tuple[str, Path, Any]:
    phase_slug = str(dataset_manifest.get("phase_slug") or "").strip()
    output_root = str(dataset_manifest.get("output_root") or "newoutput/distill")
    if not phase_slug:
        raise RuntimeError("Dataset manifest is missing phase_slug.")
    phase_paths = build_phase_paths(output_root, phase_slug)
    phase_paths.ensure()
    return phase_slug, Path(output_root), phase_paths


def _train_student(
    args: argparse.Namespace,
    *,
    train_jsonl: Path,
    val_jsonl: Path | None,
    run_tag: str,
    phase_slug: str,
    phase_paths: Any,
    dry_run: bool,
) -> dict[str, Any]:
    run_slug = slugify(f"{phase_slug}-{run_tag}")
    model_root = phase_paths.models_root / run_slug
    lora_dir = model_root / "lora"
    merged_dir = model_root / "merged"

    cmd = [
        "uv",
        "run",
        "python",
        "-m",
        "newScripts.train_student_unsloth_distill",
        "--train-jsonl",
        str(train_jsonl),
        "--eval-jsonl",
        str(val_jsonl if val_jsonl is not None else train_jsonl),
        "--max-train-samples",
        str(args.student_max_train_samples),
        "--max-eval-samples",
        str(args.student_max_eval_samples),
        "--max-seq-length",
        str(args.student_max_seq_length),
        "--packing",
        str(args.student_packing),
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
        "0.0",
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

    log_state(LOGGER, "student_train_start", run_tag=run_tag, train_jsonl=str(train_jsonl))
    run_logged_command(cmd, logger=LOGGER, dry_run=dry_run)
    train_summary_path = lora_dir / "train_summary.json"
    train_summary_payload = None
    if not dry_run and train_summary_path.exists():
        train_summary_payload = read_json(train_summary_path)
    return {
        "run_tag": run_tag,
        "train_jsonl": str(train_jsonl),
        "val_jsonl": str(val_jsonl if val_jsonl is not None else train_jsonl),
        "train_cmd": shlex.join(cmd),
        "lora_dir": str(lora_dir),
        "merged_dir": str(merged_dir),
        "train_summary_json": str(train_summary_path),
        "train_summary": train_summary_payload,
    }


def _format_metric(value: Any) -> str:
    if isinstance(value, (int, float)):
        return f"{float(value):.4f}"
    return "NA"


def _write_summary_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines: list[str] = []
    lines.append("# Distillation Phase Summary")
    lines.append("")
    lines.append(f"- Generated (UTC): `{payload.get('created_at_utc')}`")
    lines.append(f"- Phase: `{payload.get('phase_name')}`")
    lines.append(f"- Phase slug: `{payload.get('phase_slug')}`")
    lines.append(f"- Run ID: `{payload.get('pipeline_run_id')}`")
    lines.append(f"- Teacher checkpoint: `{payload.get('teacher_model')}`")
    lines.append(f"- Source comparison: `{payload.get('comparison_json')}`")
    lines.append(f"- Dataset manifest: `{payload.get('dataset_manifest_json')}`")
    lines.append("")

    filt = payload.get("filtered_subset_summary", {})
    lines.append("## Filter Summary")
    lines.append("")
    lines.append(f"- Total candidate rows: `{filt.get('total_rows')}`")
    lines.append(f"- Filter kept (teacher correct / student wrong): `{filt.get('filtered_rows')}`")
    lines.append(f"- Bucket counts before filtering: `{filt.get('bucket_counts_before')}`")
    lines.append("")

    lines.append("## Dataset Variants")
    lines.append("")
    for item in payload.get("dataset_variants", []):
        lines.append(f"- `{item.get('target_format')}`: `{item.get('path')}` (rows={item.get('rows')})")
    lines.append("")

    lines.append("## Format Eval Table")
    lines.append("")
    lines.append("| Format | Final EM | Held-out EM | Benchmark EM | ParseFail(mean) | AvgChars(mean) | EM Gap |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for row in payload.get("format_eval_summary_table", []):
        lines.append(
            "| "
            + f"{row.get('target_format')}"
            + " | "
            + f"{_format_metric(row.get('final_em'))}"
            + " | "
            + f"{_format_metric(row.get('heldout_em'))}"
            + " | "
            + f"{_format_metric(row.get('benchmark_em'))}"
            + " | "
            + f"{_format_metric(row.get('parse_fail_rate_mean'))}"
            + " | "
            + f"{_format_metric(row.get('avg_output_chars_mean'))}"
            + " | "
            + f"{_format_metric(row.get('stability_em_gap'))}"
            + " |"
        )
    lines.append("")

    winner = payload.get("winner_selection", {})
    lines.append("## Winner")
    lines.append("")
    lines.append(f"- Winner format: `{winner.get('winner_format')}`")
    lines.append(f"- Best recipe: `{winner.get('best_recipe')}`")
    lines.append(f"- Note: {winner.get('selection_note')}")
    lines.append(f"- Recipe manifest: `{payload.get('recipe_manifest_json')}`")
    lines.append(f"- Frozen eval manifest: `{payload.get('frozen_eval_manifest_json')}`")
    lines.append(f"- Preflight overlap report: `{payload.get('preflight_overlap_report_json')}`")
    lines.append("")

    lines.append("## Recipe Comparison")
    lines.append("")
    lines.append("| Recipe | Final EM | Held-out EM | Benchmark EM | ParseFail(mean) | AvgChars(mean) | Buckets | Recovery/Damage vs Pure |")
    lines.append("|---|---:|---:|---:|---:|---:|---|---|")
    for row in payload.get("recipe_comparison", {}).get("comparison_table", []):
        lines.append(
            "| "
            + f"{row.get('recipe')}"
            + " | "
            + f"{_format_metric(row.get('final_em'))}"
            + " | "
            + f"{_format_metric(row.get('heldout_em'))}"
            + " | "
            + f"{_format_metric(row.get('benchmark_em'))}"
            + " | "
            + f"{_format_metric(row.get('parse_fail_rate_mean'))}"
            + " | "
            + f"{_format_metric(row.get('avg_output_chars_mean'))}"
            + " | "
            + f"{row.get('bucket_counts')}"
            + " | "
            + f"{row.get('recovery_damage_vs_pure')}"
            + " |"
        )
    lines.append("")
    answers = payload.get("answers", {})
    lines.append(f"- Q1 (best format): `{answers.get('best_transfer_format')}`")
    lines.append(f"- Q1b (best recipe): `{answers.get('best_recipe')}`")
    lines.append(f"- Q2 (mixed beats pure): `{answers.get('mixed_beats_pure')}`")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    global LOGGER
    LOGGER = configure_logging("student_distill")
    args = parse_args()
    dataset_manifest = read_json(args.dataset_manifest_json)
    phase_slug, _, phase_paths = _phase_paths_from_manifest(dataset_manifest)
    teacher_model, run_id, experiment, parent_eval = _resolve_teacher_and_context(args, dataset_manifest)

    log_state(LOGGER, "distill_phase_start", dataset_manifest_json=args.dataset_manifest_json, phase_slug=phase_slug)

    variants_by_format = dataset_manifest.get("dataset_variants_by_format", {})
    if not isinstance(variants_by_format, dict):
        raise RuntimeError("Dataset manifest dataset_variants_by_format is invalid.")
    for target_format in TARGET_FORMATS:
        if not isinstance(variants_by_format.get(target_format), dict):
            raise RuntimeError(f"Dataset manifest missing target format: {target_format}")

    benchmark_dataset_jsonl = str(args.benchmark_dataset_jsonl or "").strip() or None
    heldout_dataset_jsonl = str(args.heldout_dataset_jsonl or "").strip() or None
    setattr(args, "benchmark_dataset_jsonl", benchmark_dataset_jsonl)
    setattr(args, "heldout_dataset_jsonl", heldout_dataset_jsonl)

    source_rows = _load_source_examples(dataset_manifest)
    source_overlap_jsonl = phase_paths.reports_root / "mining_source_overlap.jsonl"
    source_overlap_rows: list[dict[str, Any]] = []
    for i, row in enumerate(source_rows):
        source_overlap_rows.append(
            {
                IDX_FIELD: int(row.get(IDX_FIELD, i)) if str(row.get(IDX_FIELD, "")).strip() else i,
                QUESTION_FIELD: str(row.get(QUESTION_FIELD, "")).strip(),
                ANSWER_FIELD: str(row.get("gold", "") or row.get(ANSWER_FIELD, "")).strip(),
            }
        )
    write_jsonl_rows(source_overlap_jsonl, source_overlap_rows)

    final_eval_jsonl = Path(str(args.final_eval_jsonl)).expanduser() if str(args.final_eval_jsonl).strip() else (phase_paths.datasets_root / "frozen_eval_clean.jsonl")
    frozen_eval_manifest_path = phase_paths.manifests_root / "frozen_eval_manifest.json"
    frozen_eval_manifest: dict[str, Any]
    if str(args.final_eval_jsonl).strip():
        if not final_eval_jsonl.exists():
            raise FileNotFoundError(f"--final-eval-jsonl file not found: {final_eval_jsonl}")
        frozen_eval_manifest = {
            "created_at_utc": utc_now(),
            "mode": "external",
            "frozen_eval_jsonl": str(final_eval_jsonl),
            "rows": sum(1 for _ in iter_jsonl(final_eval_jsonl)),
        }
        write_json(frozen_eval_manifest_path, frozen_eval_manifest)
    elif args.rebuild_frozen_eval or not final_eval_jsonl.exists():
        if args.frozen_eval_source == "heldout":
            if heldout_dataset_jsonl:
                candidate_rows = [dict(row) for row in iter_jsonl(heldout_dataset_jsonl)]
            else:
                heldout_dataset_path = args.heldout_dataset_path or args.benchmark_dataset_path
                candidate_rows = _load_dataset_rows(
                    heldout_dataset_path,
                    args.heldout_split,
                    max_samples=max(int(args.frozen_eval_size) * 8, int(args.heldout_max_samples)),
                )
        else:
            if benchmark_dataset_jsonl:
                candidate_rows = [dict(row) for row in iter_jsonl(benchmark_dataset_jsonl)]
            else:
                candidate_rows = _load_dataset_rows(
                    args.benchmark_dataset_path,
                    args.benchmark_split,
                    max_samples=max(int(args.frozen_eval_size) * 8, int(args.benchmark_max_samples)),
                )
        frozen_info = _build_frozen_eval_jsonl(
            output_jsonl=final_eval_jsonl,
            source_rows=source_rows,
            candidate_rows=candidate_rows,
            max_rows=int(args.frozen_eval_size),
        )
        frozen_eval_manifest = {
            "created_at_utc": utc_now(),
            "mode": "generated",
            "frozen_eval_source": args.frozen_eval_source,
            "frozen_eval_size_requested": int(args.frozen_eval_size),
            "source_overlap_jsonl": str(source_overlap_jsonl),
        }
        frozen_eval_manifest.update(frozen_info)
        write_json(frozen_eval_manifest_path, frozen_eval_manifest)
    else:
        frozen_eval_manifest = read_json(frozen_eval_manifest_path) if frozen_eval_manifest_path.exists() else {
            "created_at_utc": utc_now(),
            "mode": "reused-existing",
            "frozen_eval_jsonl": str(final_eval_jsonl),
            "rows": sum(1 for _ in iter_jsonl(final_eval_jsonl)),
        }
    setattr(args, "final_eval_jsonl", str(final_eval_jsonl))
    if int(args.final_eval_max_samples) <= 0:
        setattr(args, "final_eval_max_samples", int(args.frozen_eval_size))

    shared_train_config = {
        "student_base_model": args.student_base_model,
        "student_epochs": args.student_epochs,
        "student_learning_rate": args.student_lr,
        "student_batch_size": args.student_batch_size,
        "student_grad_accum": args.student_grad_accum,
        "student_max_seq_length": args.student_max_seq_length,
        "student_packing": bool(args.student_packing),
        "student_max_train_samples": args.student_max_train_samples,
        "student_max_eval_samples": args.student_max_eval_samples,
        "student_seed": args.student_seed,
        "train_val_ratio": args.train_val_ratio,
        "split_seed": args.split_seed,
        "limit": args.limit,
        "allow_env_limit": bool(args.allow_env_limit),
        "allow_partial_eval": bool(args.allow_partial_eval),
        "overlap_threshold": float(args.overlap_threshold),
        "allow_overlap": bool(args.allow_overlap),
        "final_eval_jsonl": str(final_eval_jsonl),
    }

    preflight_named_rows: dict[str, list[dict[str, Any]]] = {
        "mining_source": source_rows,
        "distill_train_pool": [dict(row) for row in iter_jsonl(Path(str(variants_by_format[FORMAT_ANSWER_ONLY]["path"])))],
        "final_eval": [dict(row) for row in iter_jsonl(final_eval_jsonl)],
    }
    if benchmark_dataset_jsonl:
        preflight_named_rows["benchmark_eval"] = [dict(row) for row in iter_jsonl(Path(benchmark_dataset_jsonl))]
    else:
        preflight_named_rows["benchmark_eval"] = _load_dataset_rows(
            args.benchmark_dataset_path,
            args.benchmark_split,
            max_samples=int(args.benchmark_max_samples),
        )
    if heldout_dataset_jsonl:
        preflight_named_rows["heldout_eval"] = [dict(row) for row in iter_jsonl(Path(heldout_dataset_jsonl))]
    else:
        heldout_dataset_path = args.heldout_dataset_path or args.benchmark_dataset_path
        preflight_named_rows["heldout_eval"] = _load_dataset_rows(
            heldout_dataset_path,
            args.heldout_split,
            max_samples=int(args.heldout_max_samples),
        )
    preflight_overlap = compute_overlap_report(preflight_named_rows)
    preflight_overlap["threshold"] = float(args.overlap_threshold)
    preflight_overlap["allow_overlap"] = bool(args.allow_overlap)
    preflight_violations: list[dict[str, Any]] = []
    for pair in preflight_overlap.get("pairwise", []):
        left = str(pair.get("left"))
        right = str(pair.get("right"))
        if {"benchmark_eval", "heldout_eval", "final_eval"}.intersection({left, right}) and {"mining_source", "distill_train_pool"}.intersection({left, right}):
            overlap_pct = pair.get("overlap_pct_of_left") if left in {"benchmark_eval", "heldout_eval", "final_eval"} else pair.get("overlap_pct_of_right")
            if isinstance(overlap_pct, (int, float)) and float(overlap_pct) > float(args.overlap_threshold):
                preflight_violations.append(
                    {
                        "left": left,
                        "right": right,
                        "overlap_count": pair.get("overlap_count"),
                        "overlap_pct_eval": float(overlap_pct),
                    }
                )
    preflight_overlap["violations"] = preflight_violations
    preflight_overlap_path = phase_paths.reports_root / "overlap_preflight.json"
    write_json(preflight_overlap_path, preflight_overlap)
    if preflight_violations and bool(args.allow_overlap):
        LOGGER.warning(
            "Preflight overlap override enabled. violations=%s threshold=%.6f",
            len(preflight_violations),
            float(args.overlap_threshold),
        )
    if preflight_violations and not bool(args.allow_overlap):
        first = preflight_violations[0]
        raise RuntimeError(
            "Preflight overlap violation: train/source overlaps eval set above threshold. "
            f"left={first['left']} right={first['right']} overlap_count={first['overlap_count']} "
            f"overlap_pct_eval={first['overlap_pct_eval']:.6f} threshold={float(args.overlap_threshold):.6f}. "
            "Use --allow-overlap for explicit override."
        )

    split_root = phase_paths.datasets_root / "splits"
    split_root.mkdir(parents=True, exist_ok=True)
    format_runs: list[dict[str, Any]] = []
    for target_format in TARGET_FORMATS:
        variant_info = variants_by_format[target_format]
        train_jsonl = Path(str(variant_info["path"]))
        run_tag = f"format-{target_format}"
        split_info = _materialize_train_val_split(
            input_jsonl=train_jsonl,
            output_root=split_root,
            run_tag=run_tag,
            val_ratio=float(args.train_val_ratio),
            seed=int(args.split_seed),
        )
        overlap_refs = [f"mining_source={source_overlap_jsonl}"] + _make_overlap_refs(
            Path(split_info["train_jsonl"]),
            Path(split_info["val_jsonl"]),
        )
        train_info = _train_student(
            args,
            train_jsonl=Path(split_info["train_jsonl"]),
            val_jsonl=Path(split_info["val_jsonl"]),
            run_tag=run_tag,
            phase_slug=phase_slug,
            phase_paths=phase_paths,
            dry_run=args.dry_run,
        )
        eval_info = evaluate_benchmark_and_heldout(
            args,
            teacher_model=teacher_model,
            student_model=str(train_info["merged_dir"]),
            run_label=run_tag,
            phase_slug=phase_slug,
            parent_eval=parent_eval,
            experiment=experiment,
            run_id=run_id,
            dry_run=args.dry_run,
            overlap_references=overlap_refs,
        )
        format_runs.append(
            {
                "target_format": target_format,
                "dataset_path": str(train_jsonl),
                "split": split_info,
                "overlap_references": overlap_refs,
                "training": train_info,
                "evaluation": eval_info,
            }
        )

    winner = select_winner(format_runs)
    winner_format = str(winner.get("target_format") or FORMAT_ANSWER_ONLY)
    recipe_manifest_path, recipe_manifest = _build_recipe_datasets(
        phase_paths=phase_paths,
        dataset_manifest=dataset_manifest,
        winner_format=winner_format,
        seed=int(args.recipe_seed),
    )

    recipe_runs: list[dict[str, Any]] = []
    for recipe_name, recipe_info in recipe_manifest.get("recipes", {}).items():
        recipe_dataset = Path(str(recipe_info["dataset_path"]))
        run_tag = f"winner-{winner_format}-{recipe_name}"
        split_info = _materialize_train_val_split(
            input_jsonl=recipe_dataset,
            output_root=split_root,
            run_tag=run_tag,
            val_ratio=float(args.train_val_ratio),
            seed=int(args.split_seed),
        )
        overlap_refs = [f"mining_source={source_overlap_jsonl}"] + _make_overlap_refs(
            Path(split_info["train_jsonl"]),
            Path(split_info["val_jsonl"]),
        )
        train_info = _train_student(
            args,
            train_jsonl=Path(split_info["train_jsonl"]),
            val_jsonl=Path(split_info["val_jsonl"]),
            run_tag=run_tag,
            phase_slug=phase_slug,
            phase_paths=phase_paths,
            dry_run=args.dry_run,
        )
        eval_info = evaluate_benchmark_and_heldout(
            args,
            teacher_model=teacher_model,
            student_model=str(train_info["merged_dir"]),
            run_label=run_tag,
            phase_slug=phase_slug,
            parent_eval=parent_eval,
            experiment=experiment,
            run_id=run_id,
            dry_run=args.dry_run,
            overlap_references=overlap_refs,
        )
        recipe_runs.append(
            {
                "target_format": recipe_name,
                "recipe": recipe_info,
                "split": split_info,
                "overlap_references": overlap_refs,
                "training": train_info,
                "evaluation": eval_info,
            }
        )

    format_eval_table = []
    for record in format_runs:
        aggregate = record.get("evaluation", {}).get("aggregate", {})
        format_eval_table.append(
            {
                "target_format": record.get("target_format"),
                "final_em": aggregate.get("final_em"),
                "heldout_em": aggregate.get("heldout_em"),
                "benchmark_em": aggregate.get("benchmark_em"),
                "parse_fail_rate_mean": aggregate.get("parse_fail_rate_mean"),
                "avg_output_chars_mean": aggregate.get("avg_output_chars_mean"),
                "stability_em_gap": aggregate.get("stability_em_gap"),
            }
        )

    pure_recipe_record = next((r for r in recipe_runs if r.get("target_format") == "recipe_pure_target"), None)
    pure_final_cmp = None
    pure_final_rows: list[dict[str, Any]] = []
    if isinstance(pure_recipe_record, dict):
        pure_final_cmp = pure_recipe_record.get("evaluation", {}).get("final", {}).get("comparison_json")
        if isinstance(pure_final_cmp, str) and pure_final_cmp.strip():
            pure_final_rows = _load_eval_examples(pure_final_cmp)

    recipe_eval_table: list[dict[str, Any]] = []
    for record in recipe_runs:
        aggregate = record.get("evaluation", {}).get("aggregate", {})
        candidate_final_cmp = record.get("evaluation", {}).get("final", {}).get("comparison_json")
        recovery_damage = None
        if pure_final_rows and isinstance(candidate_final_cmp, str) and candidate_final_cmp.strip():
            candidate_rows = _load_eval_examples(candidate_final_cmp)
            recovery_damage = _compute_recovery_damage(pure_final_rows, candidate_rows)
        recipe_eval_table.append(
            {
                "recipe": record.get("target_format"),
                "final_em": aggregate.get("final_em"),
                "heldout_em": aggregate.get("heldout_em"),
                "benchmark_em": aggregate.get("benchmark_em"),
                "parse_fail_rate_mean": aggregate.get("parse_fail_rate_mean"),
                "avg_output_chars_mean": aggregate.get("avg_output_chars_mean"),
                "bucket_counts": record.get("recipe", {}).get("bucket_counts", {}),
                "recovery_damage_vs_pure": recovery_damage,
            }
        )

    winner_note = (
        "Selected by clean final EM (primary), held-out EM (secondary), benchmark EM (tertiary), "
        "then lower parse-failure rate, shorter outputs, and smaller held-out/benchmark EM gap."
    )
    best_recipe = select_winner(recipe_runs) if recipe_runs else None
    pure_recipe = pure_recipe_record
    mixed_best = best_recipe
    pure_final = pure_recipe.get("evaluation", {}).get("aggregate", {}).get("final_em") if pure_recipe else None
    mixed_final = mixed_best.get("evaluation", {}).get("aggregate", {}).get("final_em") if mixed_best else None
    mixed_beats_pure = (
        isinstance(mixed_final, (int, float))
        and isinstance(pure_final, (int, float))
        and float(mixed_final) > float(pure_final)
    )

    phase_summary = {
        "created_at_utc": utc_now(),
        "phase_name": dataset_manifest.get("phase_name"),
        "phase_slug": phase_slug,
        "pipeline_run_id": run_id,
        "experiment_name": experiment,
        "teacher_model": teacher_model,
        "comparison_json": dataset_manifest.get("comparison_json"),
        "dataset_manifest_json": str(Path(args.dataset_manifest_json)),
        "frozen_eval_manifest_json": str(frozen_eval_manifest_path),
        "recipe_manifest_json": str(recipe_manifest_path),
        "preflight_overlap_report_json": str(preflight_overlap_path),
        "parent_eval": parent_eval,
        "filtered_subset_summary": dataset_manifest.get("filtered_subset_summary", {}),
        "dataset_variants": dataset_manifest.get("dataset_variants", []),
        "frozen_eval_manifest": frozen_eval_manifest,
        "preflight_overlap": preflight_overlap,
        "shared_train_config": shared_train_config,
        "format_runs": format_runs,
        "format_eval_summary_table": format_eval_table,
        "recipe_runs": recipe_runs,
        "recipe_eval_summary_table": recipe_eval_table,
        "winner_selection": {
            "winner_format": winner_format,
            "selection_note": winner_note,
            "winner_metrics": winner.get("evaluation", {}).get("aggregate", {}),
            "best_recipe": best_recipe.get("target_format") if isinstance(best_recipe, dict) else None,
            "best_recipe_metrics": best_recipe.get("evaluation", {}).get("aggregate", {}) if isinstance(best_recipe, dict) else {},
        },
        "recipe_comparison": {
            "winner_format": winner_format,
            "recipe_manifest": recipe_manifest,
            "comparison_table": recipe_eval_table,
        },
        "answers": {
            "best_transfer_format": winner_format,
            "best_recipe": best_recipe.get("target_format") if isinstance(best_recipe, dict) else None,
            "mixed_beats_pure": mixed_beats_pure,
        },
    }

    summary_json_path = phase_paths.reports_root / "phase_summary.json"
    summary_md_path = phase_paths.reports_root / "phase_summary.md"
    filtered_summary_path = phase_paths.reports_root / "filtered_subset_summary.json"
    write_json(summary_json_path, phase_summary)
    write_json(filtered_summary_path, dataset_manifest.get("filtered_subset_summary", {}))
    _write_summary_markdown(summary_md_path, phase_summary)

    log_state(LOGGER, "distill_phase_end", summary_json=str(summary_json_path), summary_md=str(summary_md_path))
    LOGGER.info("DONE phase summary: %s", summary_json_path)
    LOGGER.info("DONE phase report: %s", summary_md_path)
    LOGGER.info("DONE filtered summary: %s", filtered_summary_path)


if __name__ == "__main__":
    main()
