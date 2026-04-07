#!/usr/bin/env python3
from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Any

from newScripts.common import (
    BUCKET_BOTH_CORRECT,
    BUCKET_TEACHER_CORRECT_STUDENT_WRONG,
    BUCKET_TEACHER_WRONG_STUDENT_CORRECT,
    IDX_FIELD,
    QUESTION_FIELD,
    SAMPLE_COMPARISON_KEY,
    TARGET_SOURCE_GOLD,
    TARGET_SOURCE_TEACHER,
    build_phase_paths,
    configure_logging,
    log_state,
    read_json,
    render_dataset_rows,
    safe_int,
    utc_now,
    write_json,
    write_jsonl_rows,
)

LOGGER = configure_logging("build_distill_mix")
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
    parser = argparse.ArgumentParser(description="Build mixed-bucket recipe datasets for winner format.")
    parser.add_argument("--dataset-manifest-json", type=str, required=True)
    parser.add_argument("--winner-format", choices=["answer_only", "short_rationale", "full_rationale"], required=True)
    parser.add_argument("--recipe-seed", type=int, default=42)
    return parser.parse_args()


def _load_source_rows(dataset_manifest: dict[str, Any]) -> list[dict[str, Any]]:
    comparison_json = str(dataset_manifest.get("comparison_json") or "").strip()
    if not comparison_json:
        raise RuntimeError("Dataset manifest is missing comparison_json.")
    payload = read_json(comparison_json)
    sample_cmp = payload.get(SAMPLE_COMPARISON_KEY, {})
    rows = sample_cmp.get("examples", []) if isinstance(sample_cmp, dict) else []
    if not isinstance(rows, list):
        raise RuntimeError(f"Invalid sample comparison rows in {comparison_json}")
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


def _rows_by_bucket(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {
        BUCKET_TEACHER_CORRECT_STUDENT_WRONG: [],
        BUCKET_BOTH_CORRECT: [],
        BUCKET_TEACHER_WRONG_STUDENT_CORRECT: [],
    }
    for row in rows:
        label = _bucket_label(row)
        if label in out:
            out[label].append(dict(row))
    return out


def _sample_bucket(rows: list[dict[str, Any]], count: int, seed: int, salt: str) -> list[dict[str, Any]]:
    if count <= 0:
        return []
    if len(rows) < count:
        raise RuntimeError(f"Bucket {salt} insufficient rows: requested={count} available={len(rows)}")
    rng = random.Random(f"{seed}:{salt}")
    indices = list(range(len(rows)))
    rng.shuffle(indices)
    chosen = sorted(indices[:count])
    return [dict(rows[i]) for i in chosen]


def main() -> None:
    args = parse_args()
    dataset_manifest = read_json(args.dataset_manifest_json)
    phase_slug = str(dataset_manifest.get("phase_slug") or "").strip()
    output_root = str(dataset_manifest.get("output_root") or "newoutput/distill")
    if not phase_slug:
        raise RuntimeError("Dataset manifest is missing phase_slug.")
    phase_paths = build_phase_paths(output_root, phase_slug)
    phase_paths.ensure()

    render_config = dataset_manifest.get("render_config", {})
    if not isinstance(render_config, dict):
        render_config = {}

    source_rows = _load_source_rows(dataset_manifest)
    pools = _rows_by_bucket(source_rows)
    target_total = len(pools[BUCKET_TEACHER_CORRECT_STUDENT_WRONG])
    if target_total <= 0:
        raise RuntimeError("Target bucket is empty.")

    recipe_root = phase_paths.datasets_root / "recipes" / args.winner_format
    recipe_root.mkdir(parents=True, exist_ok=True)
    recipe_records: dict[str, Any] = {}
    for recipe_name, ratios in RECIPE_DEFINITIONS.items():
        tc_sw_count = int(round(target_total * float(ratios[BUCKET_TEACHER_CORRECT_STUDENT_WRONG])))
        bc_count = int(round(target_total * float(ratios[BUCKET_BOTH_CORRECT])))
        tw_sc_count = max(0, target_total - tc_sw_count - bc_count)
        requested = {
            BUCKET_TEACHER_CORRECT_STUDENT_WRONG: tc_sw_count,
            BUCKET_BOTH_CORRECT: bc_count,
            BUCKET_TEACHER_WRONG_STUDENT_CORRECT: tw_sc_count,
        }
        rendered_rows: list[dict[str, Any]] = []
        source_ids: list[int] = []
        per_bucket_counts: dict[str, int] = {}
        for bucket_name in (
            BUCKET_TEACHER_CORRECT_STUDENT_WRONG,
            BUCKET_BOTH_CORRECT,
            BUCKET_TEACHER_WRONG_STUDENT_CORRECT,
        ):
            needed = requested[bucket_name]
            if needed <= 0:
                per_bucket_counts[bucket_name] = 0
                continue
            sampled = _sample_bucket(pools[bucket_name], needed, args.recipe_seed, f"{recipe_name}:{bucket_name}")
            target_source = TARGET_SOURCE_TEACHER if bucket_name == BUCKET_TEACHER_CORRECT_STUDENT_WRONG else TARGET_SOURCE_GOLD
            rows_for_bucket, bucket_ids = render_dataset_rows(
                sampled,
                target_format=args.winner_format,
                target_source=target_source,
                short_max_sentences=int(render_config.get("short_rationale_max_sentences", 3)),
                short_max_chars=int(render_config.get("short_rationale_max_chars", 360)),
                max_answer_chars=int(render_config.get("max_answer_chars", 2000)),
            )
            if len(rows_for_bucket) != needed:
                raise RuntimeError(
                    f"Rendered count mismatch recipe={recipe_name} bucket={bucket_name}: requested={needed} rendered={len(rows_for_bucket)}"
                )
            rendered_rows.extend(rows_for_bucket)
            source_ids.extend([safe_int(x, -1) for x in bucket_ids])
            per_bucket_counts[bucket_name] = len(rows_for_bucket)

        output_jsonl = recipe_root / f"{recipe_name}.jsonl"
        write_jsonl_rows(output_jsonl, rendered_rows)
        recipe_records[recipe_name] = {
            "dataset_path": str(output_jsonl),
            "rows": len(rendered_rows),
            "bucket_counts": per_bucket_counts,
            "target_ratios": ratios,
            "source_ids": source_ids,
        }

    manifest = {
        "created_at_utc": utc_now(),
        "phase_slug": phase_slug,
        "dataset_manifest_json": str(Path(args.dataset_manifest_json)),
        "winner_format": args.winner_format,
        "recipe_seed": int(args.recipe_seed),
        "target_total_rows": target_total,
        "bucket_pool_sizes": {
            BUCKET_TEACHER_CORRECT_STUDENT_WRONG: len(pools[BUCKET_TEACHER_CORRECT_STUDENT_WRONG]),
            BUCKET_BOTH_CORRECT: len(pools[BUCKET_BOTH_CORRECT]),
            BUCKET_TEACHER_WRONG_STUDENT_CORRECT: len(pools[BUCKET_TEACHER_WRONG_STUDENT_CORRECT]),
        },
        "recipes": recipe_records,
    }
    manifest_path = phase_paths.manifests_root / f"bucket_recipe_manifest_{args.winner_format}.json"
    write_json(manifest_path, manifest)
    log_state(
        LOGGER,
        "bucket_recipe_build_end",
        winner_format=args.winner_format,
        manifest_json=str(manifest_path),
        target_total=target_total,
    )
    LOGGER.info("DONE recipe manifest: %s", manifest_path)


if __name__ == "__main__":
    main()
