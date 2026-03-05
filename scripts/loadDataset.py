from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Optional
from typing import Any

from datasets import concatenate_datasets, load_dataset
from datasets.exceptions import ExpectedMoreSplitsError

from Cdatasets.tokenizer import load_tokenizer
from utils.packed_dataset_builder import PackedDatasetBuilder

DEFAULT_TOKENIZER_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
DEFAULT_OUTPUT_DIR = "/tmp/vertex-bins"
DEFAULT_BIN_NAME = "data.bin"
DEFAULT_DATASET_NAME = "default"

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


def _parse_bool_flag(value: str | int | bool) -> int:
    if isinstance(value, bool):
        return 1 if value else 0
    if isinstance(value, int):
        return 1 if value != 0 else 0
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "t", "yes", "y", "on"}:
        return 1
    if normalized in {"0", "false", "f", "no", "n", "off"}:
        return 0
    raise argparse.ArgumentTypeError(f"Invalid boolean flag value: {value!r}")


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
    parser.add_argument(
        "--enable_bucket",
        "--enable-bucket",
        type=_parse_bool_flag,
        default=_parse_bool_flag(os.environ.get("ENABLE_BUCKET", "1")),
        choices=[0, 1],
        help="Enable GCS dataset cache sync (1=enabled, 0=disabled).",
    )
    parser.add_argument(
        "--bucket_uri",
        type=str,
        default=os.environ.get("BUCKET_URI", ""),
        help="Bucket URI, e.g. gs://my-bucket or gs://my-bucket/some/prefix",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=os.environ.get("DATASET_NAME", DEFAULT_DATASET_NAME),
        help="Dataset cache name used in default bucket path datasets/{dataset_name}.",
    )
    parser.add_argument(
        "--bucket_dataset_path",
        type=str,
        default=os.environ.get("BUCKET_DATASET_PATH", ""),
        help="Optional custom bucket path override (relative path inside the bucket).",
    )
    return parser.parse_args()


def _parse_gcs_uri(uri: str) -> tuple[str, str]:
    raw = uri.strip()
    if not raw.startswith("gs://"):
        raise ValueError(f"Expected GCS URI starting with gs://, got {uri!r}")
    tail = raw[5:].strip("/")
    if not tail:
        raise ValueError(f"GCS URI is missing bucket name: {uri!r}")
    if "/" not in tail:
        return tail, ""
    bucket, prefix = tail.split("/", 1)
    return bucket, prefix.strip("/")


def _join_gcs_prefix(*parts: str) -> str:
    cleaned = [p.strip("/") for p in parts if p and p.strip("/")]
    return "/".join(cleaned)


def _resolve_bucket_target(args: argparse.Namespace) -> Optional[tuple[str, str]]:
    candidates = [
        str(getattr(args, "bucket_uri", "") or "").strip(),
        str(os.environ.get("AIP_MODEL_DIR", "") or "").strip(),
        str(os.environ.get("AIP_CHECKPOINT_DIR", "") or "").strip(),
    ]
    bucket_uri = next((c for c in candidates if c.startswith("gs://")), "")
    if not bucket_uri:
        return None

    bucket_name, base_prefix = _parse_gcs_uri(bucket_uri)
    custom_path = str(getattr(args, "bucket_dataset_path", "") or "").strip()
    if custom_path:
        dataset_prefix = custom_path.strip("/")
    else:
        dataset_name = str(getattr(args, "dataset_name", DEFAULT_DATASET_NAME) or "").strip() or DEFAULT_DATASET_NAME
        dataset_prefix = f"datasets/{dataset_name}"
    return bucket_name, _join_gcs_prefix(base_prefix, dataset_prefix)


def _download_cache_from_bucket(
    args: argparse.Namespace,
    bin_path: Path,
    idx_path: Path,
    summary_path: Path,
) -> bool:
    if int(getattr(args, "enable_bucket", 1)) != 1:
        return False

    target = _resolve_bucket_target(args)
    if target is None:
        print("[CACHE] Bucket sync enabled but no GCS URI found. Skipping remote download.")
        return False

    try:
        from google.cloud import storage
    except Exception as exc:
        print(f"[CACHE] google-cloud-storage unavailable ({exc}); skipping remote download.")
        return False

    bucket_name, prefix = target
    bin_blob_name = _join_gcs_prefix(prefix, bin_path.name)
    idx_blob_name = _join_gcs_prefix(prefix, idx_path.name)
    summary_blob_name = _join_gcs_prefix(prefix, summary_path.name)

    try:
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        bin_blob = bucket.blob(bin_blob_name)
        idx_blob = bucket.blob(idx_blob_name)

        if not bin_blob.exists() or not idx_blob.exists():
            print(
                f"[CACHE] No remote cache at gs://{bucket_name}/{prefix} "
                f"(need {bin_path.name} + {idx_path.name})."
            )
            return False

        bin_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"[CACHE] Downloading {bin_path.name} from gs://{bucket_name}/{bin_blob_name}")
        bin_blob.download_to_filename(str(bin_path))
        print(f"[CACHE] Downloading {idx_path.name} from gs://{bucket_name}/{idx_blob_name}")
        idx_blob.download_to_filename(str(idx_path))

        summary_blob = bucket.blob(summary_blob_name)
        if summary_blob.exists():
            print(f"[CACHE] Downloading summary from gs://{bucket_name}/{summary_blob_name}")
            summary_blob.download_to_filename(str(summary_path))

        print(f"[CACHE] Reused packed dataset from gs://{bucket_name}/{prefix}")
        return True
    except Exception as exc:
        print(f"[CACHE] Remote download failed ({exc}); proceeding with local build.")
        return False


def _upload_cache_to_bucket(
    args: argparse.Namespace,
    bin_path: Path,
    idx_path: Path,
    summary_path: Path,
) -> None:
    if int(getattr(args, "enable_bucket", 1)) != 1:
        return

    target = _resolve_bucket_target(args)
    if target is None:
        print("[CACHE] Bucket sync enabled but no GCS URI found. Skipping remote upload.")
        return

    try:
        from google.cloud import storage
    except Exception as exc:
        print(f"[CACHE] google-cloud-storage unavailable ({exc}); skipping remote upload.")
        return

    bucket_name, prefix = target
    files = [bin_path, idx_path, summary_path]

    try:
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        for path in files:
            if not path.exists():
                continue
            blob_name = _join_gcs_prefix(prefix, path.name)
            print(f"[CACHE] Uploading {path.name} to gs://{bucket_name}/{blob_name}")
            bucket.blob(blob_name).upload_from_filename(str(path))
        print(f"[CACHE] Uploaded packed dataset cache to gs://{bucket_name}/{prefix}")
    except Exception as exc:
        print(f"[CACHE] Remote upload failed ({exc}); keeping local files only.")


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
    print(f"[LOAD] output_dir={output_dir}")

    target_bin_path = output_dir / args.output_bin_name
    target_idx_path = output_dir / (Path(args.output_bin_name).stem + ".idx")
    summary_path = output_dir / "dataset_build_summary.json"

    if target_bin_path.exists() and target_idx_path.exists():
        print(f"[CACHE] Found local packed dataset: {target_bin_path} and {target_idx_path}. Reusing.")
        if summary_path.exists():
            summary = json.loads(summary_path.read_text())
        else:
            summary = {
                "bin_path": str(target_bin_path),
                "idx_path": str(target_idx_path),
                "cache_source": "local",
            }
            summary_path.write_text(json.dumps(summary, indent=2))
        _upload_cache_to_bucket(args, target_bin_path, target_idx_path, summary_path)
        print("[DONE] Reused existing packed dataset.")
        return summary

    if _download_cache_from_bucket(args, target_bin_path, target_idx_path, summary_path):
        if summary_path.exists():
            summary = json.loads(summary_path.read_text())
        else:
            summary = {
                "bin_path": str(target_bin_path),
                "idx_path": str(target_idx_path),
                "cache_source": "bucket",
            }
            summary_path.write_text(json.dumps(summary, indent=2))
        print("[DONE] Reused packed dataset from bucket.")
        return summary

    mixture = load_dataset_mixture(args.datasets_json)
    print(f"[LOAD] datasets_in_mixture={len(mixture)}")

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
        "cache_source": "built",
    }
    summary_path.write_text(json.dumps(summary, indent=2))
    _upload_cache_to_bucket(args, target_bin_path, target_idx_path, summary_path)

    print("[DONE] Packed dataset build complete.") 
    print(f"[DONE] total_sequences={summary['total_sequences']:,}")
    print(f"[DONE] --output_bin_name={summary_path}")
    return summary


def main() -> None:
    args = parse_args()
    build_single_packed_bin(args)


if __name__ == "__main__":
    main()
