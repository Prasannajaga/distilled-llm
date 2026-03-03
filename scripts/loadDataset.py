from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from datasets import concatenate_datasets, load_dataset
from datasets.exceptions import ExpectedMoreSplitsError

from Cdatasets.tokenizer import load_tokenizer
from utils.packed_dataset_builder import PackedDatasetBuilder

DEFAULT_TOKENIZER_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
DEFAULT_OUTPUT_DIR = "/tmp/vertex-bins"
DEFAULT_BIN_NAME = "data.bin"

# Simple mixture definition: each item supports
# name (required), subset (optional), split (optional), text_column (required), data_files (optional).
DEFAULT_DATASET_MIXTURE: list[dict[str, Any]] = [
    {
        "name": "keirp/open-web-math-hq-dev",
        "split": "train",
        "text_column": "text",
        "data_files": ["data/train-00000-of-00044-9f55eda5b15e5628.parquet"],
    },
    {
        "name": "DKYoon/SlimPajama-6B",
        "split": "train",
        "text_column": "text",
        "data_files": ["data/test-00000-of-00001-9f769cf7ce219017.parquet"],
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Load a dataset mixture and build one packed bin file."
    )
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--output_bin_name", type=str, default=DEFAULT_BIN_NAME)
    parser.add_argument("--datasets_json", type=str, default=None)
    parser.add_argument("--block_size", type=int, default=512)
    parser.add_argument("--num_workers", type=int, default=int(os.cpu_count() or 4))
    parser.add_argument("--pack_batch_size", type=int, default=1024)
    parser.add_argument("--token_dtype", type=str, default="auto", choices=["auto", "uint16", "uint32"])
    parser.add_argument("--tokenizer_model", type=str, default=DEFAULT_TOKENIZER_MODEL)
    return parser.parse_args()


def load_dataset_mixture(datasets_json: str | None) -> list[dict[str, Any]]:
    if not datasets_json:
        return list(DEFAULT_DATASET_MIXTURE)

    source = Path(datasets_json)
    raw = source.read_text() if source.exists() else datasets_json
    parsed = json.loads(raw)
    if not isinstance(parsed, list):
        raise ValueError("datasets_json must be a JSON array of dataset objects.")

    normalized: list[dict[str, Any]] = []
    for index, item in enumerate(parsed):
        if not isinstance(item, dict):
            raise ValueError(f"Dataset entry at index {index} must be an object.")
        if "name" not in item:
            raise ValueError(f"Dataset entry at index {index} is missing 'name'.")
        if "text_column" not in item:
            raise ValueError(f"Dataset entry at index {index} is missing 'text_column'.")
        normalized.append(item)
    return normalized


def _load_one_dataset(spec: dict[str, Any], output_dir: Path):
    dataset_name = str(spec["name"])
    subset = spec.get("subset")
    split = str(spec.get("split", "train"))
    text_column = str(spec["text_column"])

    reserved_keys = {"name", "subset", "split", "text_column"}
    load_kwargs = {k: v for k, v in spec.items() if k not in reserved_keys}
    if "data_files" in load_kwargs and "verification_mode" not in load_kwargs:
        # If we load only selected shard files, split metadata checks often fail.
        load_kwargs["verification_mode"] = "no_checks"

    print(
        f"[LOAD] dataset={dataset_name} subset={subset} split={split} "
        f"text_column={text_column}"
    )
    if "data_files" in load_kwargs:
        print(f"[LOAD] data_files={load_kwargs['data_files']}")

    try:
        hf_dataset = load_dataset(
            dataset_name,
            subset,
            split=split,
            cache_dir=str(output_dir),
            **load_kwargs,
        )
    except ExpectedMoreSplitsError:
        retry_kwargs = dict(load_kwargs)
        retry_kwargs["verification_mode"] = "no_checks"
        print(
            f"[LOAD] Retrying {dataset_name} with verification_mode=no_checks "
            "because only partial split files were provided."
        )
        hf_dataset = load_dataset(
            dataset_name,
            subset,
            split=split,
            cache_dir=str(output_dir),
            **retry_kwargs,
        )
    if text_column not in hf_dataset.column_names:
        raise ValueError(
            f"Column '{text_column}' not found in {dataset_name}. "
            f"Available columns: {hf_dataset.column_names}"
        )

    # Keep only the text column and normalize to "text" so all datasets can be concatenated.
    reduced = hf_dataset.select_columns([text_column])
    if text_column != "text":
        reduced = reduced.rename_column(text_column, "text")
    return reduced


def build_single_packed_bin(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    mixture = load_dataset_mixture(args.datasets_json)
    print(f"[LOAD] datasets_in_mixture={len(mixture)}")
    print(f"[LOAD] output_dir={output_dir}")

    loaded_parts = []
    dataset_row_counts: list[dict[str, Any]] = []
    for spec in mixture:
        part = _load_one_dataset(spec, output_dir)
        loaded_parts.append(part)
        dataset_row_counts.append(
            {
                "name": str(spec.get("name", "")),
                "subset": spec.get("subset"),
                "split": str(spec.get("split", "train")),
                "text_column": str(spec.get("text_column", "text")),
                "rows": int(len(part)),
            }
        )
    merged_dataset = loaded_parts[0] if len(loaded_parts) == 1 else concatenate_datasets(loaded_parts)
    print(f"[LOAD] merged_rows={len(merged_dataset):,}")

    tokenizer = load_tokenizer(args.tokenizer_model)
    builder = PackedDatasetBuilder(
        dataset_name="merged_dataset",
        tokenizer=tokenizer,
        block_size=args.block_size,
        output_path=str(output_dir),
        split="train",
        text_column="text",
        num_workers=args.num_workers,
        batch_size=args.pack_batch_size,
        token_dtype=args.token_dtype,
    )
    builder.build_from_hf_dataset(merged_dataset)

    default_bin_path = output_dir / "data.bin"
    default_idx_path = output_dir / "data.idx"
    target_bin_path = output_dir / args.output_bin_name
    target_idx_path = output_dir / (Path(args.output_bin_name).stem + ".idx")

    if target_bin_path != default_bin_path:
        if target_bin_path.exists():
            target_bin_path.unlink()
        default_bin_path.rename(target_bin_path)
    if target_idx_path != default_idx_path:
        if target_idx_path.exists():
            target_idx_path.unlink()
        default_idx_path.rename(target_idx_path)

    summary = {
        "bin_path": str(target_bin_path),
        "idx_path": str(target_idx_path),
        "total_tokens": builder.total_tokens,
        "total_sequences": builder.total_sequences,
        "total_dataset_samples": int(len(merged_dataset)),
        "dataset_row_counts": dataset_row_counts,
        "datasets": mixture,
    }
    summary_path = output_dir / "dataset_build_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))

    print("[DONE] Packed dataset build complete.") 
    print(f"[DONE] total_sequences={summary['total_sequences']:,}")
    print(f"[DONE] --output_bin_name={summary_path}")
    return summary


def main() -> None:
    args = parse_args()
    build_single_packed_bin(args)


if __name__ == "__main__":
    main()
