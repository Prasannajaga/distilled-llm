#!/usr/bin/env python3
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from newScripts.common import (
    EXAMPLES_KEY,
    FORMAT_ANSWER_ONLY,
    SAMPLE_COMPARISON_KEY,
    TARGET_FORMATS,
    TARGET_SOURCE_TEACHER,
    build_phase_paths,
    compute_bucket_counts,
    configure_logging,
    filter_teacher_correct_student_wrong,
    log_state,
    read_json,
    render_dataset_rows,
    slugify,
    utc_now,
    write_json,
    write_jsonl_rows,
)

LOGGER = configure_logging("build_distill_variants")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build teacher-target distillation dataset variants and a manifest.")
    parser.add_argument("--comparison-json", type=str, required=True)
    parser.add_argument("--output-root", type=str, default="newoutput/distill")
    parser.add_argument("--phase-name", type=str, default="next-distill-phase")
    parser.add_argument("--pipeline-run-id", type=str, default="")
    parser.add_argument("--max-filtered-rows", type=int, default=0, help="<=0 means all filtered rows.")
    parser.add_argument("--max-answer-chars", type=int, default=2000)
    parser.add_argument("--short-rationale-max-sentences", type=int, default=3)
    parser.add_argument("--short-rationale-max-chars", type=int, default=360)
    return parser.parse_args()


def _resolve_run_id(run_id: str) -> str:
    clean = str(run_id or "").strip()
    if clean:
        return clean
    return datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")


def _load_examples(comparison_json: str) -> list[dict[str, Any]]:
    payload = read_json(comparison_json)
    sample_comparison = payload.get(SAMPLE_COMPARISON_KEY, {})
    rows = sample_comparison.get(EXAMPLES_KEY, []) if isinstance(sample_comparison, dict) else []
    if not isinstance(rows, list):
        raise RuntimeError(f"Invalid {SAMPLE_COMPARISON_KEY}/{EXAMPLES_KEY} in {comparison_json}")
    return [dict(row) for row in rows]


def main() -> None:
    args = parse_args()
    run_id = _resolve_run_id(args.pipeline_run_id)
    phase_slug = slugify(f"{args.phase_name}-{run_id}")
    phase_paths = build_phase_paths(args.output_root, phase_slug)
    phase_paths.ensure()

    log_state(LOGGER, "variant_build_start", comparison_json=args.comparison_json, phase_slug=phase_slug)
    rows = _load_examples(args.comparison_json)
    bucket_counts = compute_bucket_counts(rows)
    filtered_rows = filter_teacher_correct_student_wrong(rows, max_rows=int(args.max_filtered_rows))
    if not filtered_rows:
        raise RuntimeError("Filtered subset is empty (teacher correct / student wrong).")

    filtered_summary = {
        "total_rows": len(rows),
        "filtered_rows": len(filtered_rows),
        "bucket_counts_before": bucket_counts,
        "excluded_counts": {
            "both correct": bucket_counts.get("both correct", 0),
            "both wrong": bucket_counts.get("both wrong", 0),
            "teacher wrong / student correct": bucket_counts.get("teacher wrong / student correct", 0),
        },
    }

    render_config = {
        "max_filtered_rows": int(args.max_filtered_rows),
        "max_answer_chars": int(args.max_answer_chars),
        "short_rationale_max_sentences": int(args.short_rationale_max_sentences),
        "short_rationale_max_chars": int(args.short_rationale_max_chars),
    }

    dataset_variants: list[dict[str, Any]] = []
    dataset_variants_by_format: dict[str, dict[str, Any]] = {}
    variant_source_ids: dict[str, list[int]] = {}
    for target_format in TARGET_FORMATS:
        rows_for_format, source_ids = render_dataset_rows(
            filtered_rows,
            target_format=target_format,
            target_source=TARGET_SOURCE_TEACHER,
            short_max_sentences=render_config["short_rationale_max_sentences"],
            short_max_chars=render_config["short_rationale_max_chars"],
            max_answer_chars=render_config["max_answer_chars"],
        )
        if not rows_for_format:
            raise RuntimeError(f"No rows remained after rendering format={target_format}.")
        dataset_path = phase_paths.datasets_root / f"{target_format}.jsonl"
        write_jsonl_rows(dataset_path, rows_for_format)
        record = {
            "target_format": target_format,
            "target_source": TARGET_SOURCE_TEACHER,
            "path": str(dataset_path),
            "rows": len(rows_for_format),
        }
        dataset_variants.append(record)
        dataset_variants_by_format[target_format] = record
        variant_source_ids[target_format] = source_ids
        log_state(LOGGER, "variant_written", target_format=target_format, rows=len(rows_for_format), path=str(dataset_path))

    reference_ids = variant_source_ids[FORMAT_ANSWER_ONLY]
    for target_format in TARGET_FORMATS[1:]:
        if variant_source_ids[target_format] != reference_ids:
            raise RuntimeError("Dataset variants do not share identical membership/order.")

    manifest = {
        "created_at_utc": utc_now(),
        "phase_name": args.phase_name,
        "phase_slug": phase_slug,
        "pipeline_run_id": run_id,
        "output_root": str(Path(args.output_root)),
        "phase_root": str(phase_paths.phase_root),
        "paths": {
            "datasets_root": str(phase_paths.datasets_root),
            "manifests_root": str(phase_paths.manifests_root),
            "models_root": str(phase_paths.models_root),
            "reports_root": str(phase_paths.reports_root),
        },
        "comparison_json": str(Path(args.comparison_json)),
        "render_config": render_config,
        "filtered_subset_summary": filtered_summary,
        "filtered_source_ids": reference_ids,
        "dataset_variants": dataset_variants,
        "dataset_variants_by_format": dataset_variants_by_format,
    }

    manifest_path = phase_paths.manifests_root / "dataset_manifest.json"
    filtered_summary_path = phase_paths.manifests_root / "filtered_subset_summary.json"
    write_json(manifest_path, manifest)
    write_json(filtered_summary_path, filtered_summary)
    log_state(LOGGER, "variant_build_end", manifest_json=str(manifest_path), filtered_rows=len(reference_ids))
    LOGGER.info("DONE dataset manifest: %s", manifest_path)


if __name__ == "__main__":
    main()
