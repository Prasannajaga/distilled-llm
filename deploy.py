import argparse
import shlex
import subprocess
import sys
import warnings
from datetime import datetime
from pathlib import Path

# Suppress the FutureWarnings concerning Python 3.10 End of Life 
# from the Google Cloud SDK, to keep stdout clean.
warnings.filterwarnings("ignore", category=FutureWarning, module="google.*")

from google.cloud import aiplatform
from google.cloud import storage


class FileArgParser(argparse.ArgumentParser):
    """Argument parser that supports @args files with comments."""

    def convert_arg_line_to_args(self, arg_line: str):
        line = arg_line.strip()
        if not line or line.startswith("#"):
            return []
        return shlex.split(line)


def parse_args() -> argparse.Namespace:
    parser = FileArgParser(
        description="Build package and submit Vertex AI Custom Job",
        fromfile_prefix_chars="@",
    )
    parser.add_argument("--project_id", type=str, required=True, help="GCP Project ID")
    parser.add_argument("--region", type=str, required=True, help="GCP region, e.g. us-central1")
    parser.add_argument("--bucket_uri", type=str, required=True, help="GCS bucket URI, e.g. gs://my-bucket")

    parser.add_argument("--display_name", type=str, default=None, help="Vertex job display name")
    parser.add_argument("--base_output_dir", type=str, default=None, help="Vertex base output directory in GCS")
    parser.add_argument("--staging_bucket", type=str, default=None, help="Vertex staging bucket URI")

    parser.add_argument("--machine_type", type=str, default="g2-standard-24")
    parser.add_argument("--accelerator_type", type=str, default="NVIDIA_L4")
    parser.add_argument("--accelerator_count", type=int, default=2)
    parser.add_argument("--replica_count", type=int, default=1)
    parser.add_argument("--boot_disk_size", type=int, default=100, help="Boot disk size in GB")
    parser.add_argument("--train_image", type=str, default="us-docker.pkg.dev/vertex-ai/training/pytorch-gpu.2-4.py310:latest")

    parser.add_argument("--train_module", type=str, default="scripts.run_step_train", help="Python module launched by Vertex (default runs step-train.sh)")
    parser.add_argument("--nproc_per_node", type=int, default=None, help="torchrun local processes for step-train.sh; default=accelerator_count")
    parser.add_argument("--local_ckpt_dir", type=str, default="/outputs/mini-code-v1")
    parser.add_argument("--train_log_level", type=str, default="INFO")
    parser.add_argument("--train_progress_bar", type=int, default=0, choices=[0, 1], help="Enable (1) or disable (0) tqdm progress bars")
    parser.add_argument("--data_bin_dir", type=str, default="/tmp/vertex-bins", help="Directory to write the packed bin file")
    parser.add_argument("--data_bin_name", type=str, default="data.bin", help="Output packed bin file name")
    parser.add_argument("--block_size", type=int, default=512, help="Token block size for packing")
    parser.add_argument("--datasets_json", type=str, default=None, help="Dataset mixture JSON for scripts.loadDataset")
    parser.add_argument("--data_num_workers", type=int, default=None, help="Workers for dataset packing")
    parser.add_argument("--data_pack_batch_size", type=int, default=1024, help="Batch size for dataset packing tokenization")
    parser.add_argument("--tokenizer_model", type=str, default="TinyLlama/TinyLlama-1.1B-Chat-v1.0", help="Tokenizer model name used in scripts.loadDataset")

    # Training hyperparameters passed via env -> scripts.train_vertex args defaults.
    parser.add_argument("--train_device", type=str, default="cuda")
    parser.add_argument("--train_seed", type=int, default=42)
    parser.add_argument("--train_num_workers", type=int, default=4)
    parser.add_argument("--n_layer", type=int, default=12)
    parser.add_argument("--n_embd", type=int, default=768)
    parser.add_argument("--n_head", type=int, default=12)
    parser.add_argument("--train_batch_size", type=int, default=8)
    parser.add_argument("--eval_batch_size", type=int, default=4)
    parser.add_argument("--grad_accum_steps", type=int, default=8)
    parser.add_argument("--validation_batch_count", type=int, default=20)
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--optimizer", type=str, default="adamw")
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--eps", type=float, default=1e-8)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--grad_clip_norm", type=float, default=1.0)
    parser.add_argument("--use_amp", type=int, default=1, choices=[0, 1])
    parser.add_argument("--amp_dtype", type=str, default="bfloat16")
    parser.add_argument("--enable_gradient_checkpointing", type=int, default=1, choices=[0, 1])
    parser.add_argument("--total_steps", type=int, default=4000)
    parser.add_argument("--warmup_steps", type=int, default=400)
    parser.add_argument("--ckpt_interval_steps", type=int, default=500)
    parser.add_argument("--save_optimizer_state", type=int, default=1, choices=[0, 1])
    parser.add_argument("--log_interval_steps", type=int, default=100)
    parser.add_argument("--async_run", action="store_true", help="Submit job and return immediately")
    return parser.parse_args()


def _bucket_name(bucket_uri: str) -> str:
    return bucket_uri.replace("gs://", "").strip("/").split("/")[0]


def _resolve_datasets_json_arg(value: str | None) -> str | None:
    """Return a JSON string for DATASETS_JSON.

    If `value` points to a local file, inline file content so Vertex workers
    can parse it even when the original local path is unavailable.
    """
    if value is None:
        return None
    raw = value.strip()
    if not raw:
        return None

    candidate = Path(raw).expanduser()
    if candidate.exists() and candidate.is_file():
        return candidate.read_text(encoding="utf-8")
    return raw


def build_and_upload_package(bucket_uri: str, project_id: str) -> str:
    """Build source distribution and upload it to GCS."""
    print("1. Building source distribution...")
    subprocess.check_call([sys.executable, "setup.py", "sdist", "--formats=gztar"])

    dist_dir = Path("dist")
    artifacts = sorted(dist_dir.glob("*.tar.gz"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not artifacts:
        raise FileNotFoundError("No .tar.gz artifact found in dist/")

    package_path = artifacts[0]
    package_file = package_path.name

    print(f"2. Uploading {package_file} to Cloud Storage...")
    storage_client = storage.Client(project=project_id)
    bucket_name = _bucket_name(bucket_uri)
    bucket = storage_client.bucket(bucket_name)

    gcs_blob_path = f"packages/{package_file}"
    blob = bucket.blob(gcs_blob_path)
    blob.upload_from_filename(str(package_path))

    package_gcs_uri = f"gs://{bucket_name}/{gcs_blob_path}"
    print(f"   Uploaded package to: {package_gcs_uri}")
    return package_gcs_uri


def main() -> None:
    args = parse_args()
    package_gcs_uri = build_and_upload_package(args.bucket_uri, args.project_id)

    staging_bucket = args.staging_bucket or args.bucket_uri
    base_output_dir = args.base_output_dir or f"{args.bucket_uri.rstrip('/')}/training_logs"
    display_name = args.display_name or f"distilled-llm-train-{datetime.utcnow().strftime('%Y%m%d-%H%M%S')}"
    accelerator_count = max(0, int(args.accelerator_count))
    nproc_per_node = args.nproc_per_node if args.nproc_per_node is not None else max(1, accelerator_count)

    aiplatform.init(project=args.project_id, location=args.region, staging_bucket=staging_bucket)

    print("3. Submitting Vertex AI Custom Job...")
    print(f"   display_name={display_name}")
    print(f"   module={args.train_module} | nproc_per_node={nproc_per_node}")
    print(f"   machine={args.machine_type} | accel={args.accelerator_type}:{args.accelerator_count}")

    python_package_spec = {
        "executor_image_uri": args.train_image,
        "package_uris": [package_gcs_uri],
        "python_module": args.train_module,
        "args": [],
        "env": [
            {"name": "LOCAL_CKPT_DIR", "value": args.local_ckpt_dir},
            {"name": "TRAIN_LOG_LEVEL", "value": args.train_log_level.upper()},
            {"name": "TRAIN_PROGRESS_BAR", "value": str(args.train_progress_bar)},
            {"name": "NPROC_PER_NODE", "value": str(nproc_per_node)},
            {"name": "DATA_BIN_DIR", "value": args.data_bin_dir},
            {"name": "DATA_BIN_NAME", "value": args.data_bin_name},
            {"name": "BLOCK_SIZE", "value": str(args.block_size)},
            {"name": "DATA_PACK_BATCH_SIZE", "value": str(args.data_pack_batch_size)},
            {"name": "TOKENIZER_MODEL", "value": args.tokenizer_model},
            {"name": "TRAIN_DEVICE", "value": args.train_device},
            {"name": "TRAIN_SEED", "value": str(args.train_seed)},
            {"name": "TRAIN_NUM_WORKERS", "value": str(args.train_num_workers)},
            {"name": "TRAIN_N_LAYER", "value": str(args.n_layer)},
            {"name": "TRAIN_N_EMBD", "value": str(args.n_embd)},
            {"name": "TRAIN_N_HEAD", "value": str(args.n_head)},
            {"name": "TRAIN_BLOCK_SIZE", "value": str(args.block_size)},
            {"name": "TRAIN_TRAIN_BATCH_SIZE", "value": str(args.train_batch_size)},
            {"name": "TRAIN_EVAL_BATCH_SIZE", "value": str(args.eval_batch_size)},
            {"name": "TRAIN_GRAD_ACCUM_STEPS", "value": str(args.grad_accum_steps)},
            {"name": "TRAIN_VALIDATION_BATCH_COUNT", "value": str(args.validation_batch_count)},
            {"name": "TRAIN_VAL_RATIO", "value": str(args.val_ratio)},
            {"name": "TRAIN_OPTIMIZER", "value": args.optimizer},
            {"name": "TRAIN_LR", "value": str(args.lr)},
            {"name": "TRAIN_EPS", "value": str(args.eps)},
            {"name": "TRAIN_WEIGHT_DECAY", "value": str(args.weight_decay)},
            {"name": "TRAIN_GRAD_CLIP_NORM", "value": str(args.grad_clip_norm)},
            {"name": "TRAIN_USE_AMP", "value": str(args.use_amp)},
            {"name": "TRAIN_AMP_DTYPE", "value": args.amp_dtype},
            {"name": "TRAIN_ENABLE_GRADIENT_CHECKPOINTING", "value": str(args.enable_gradient_checkpointing)},
            {"name": "TRAIN_TOTAL_STEPS", "value": str(args.total_steps)},
            {"name": "TRAIN_WARMUP_STEPS", "value": str(args.warmup_steps)},
            {"name": "TRAIN_CKPT_INTERVAL_STEPS", "value": str(args.ckpt_interval_steps)},
            {"name": "TRAIN_SAVE_OPTIMIZER_STATE", "value": str(args.save_optimizer_state)},
            {"name": "TRAIN_LOG_INTERVAL_STEPS", "value": str(args.log_interval_steps)},
            # Suppress noisy pip install warnings that appear as ERROR in Vertex log UI.
            {"name": "PIP_NO_WARN_SCRIPT_LOCATION", "value": "1"},
            {"name": "PIP_ROOT_USER_ACTION", "value": "ignore"},
            {"name": "PIP_DISABLE_PIP_VERSION_CHECK", "value": "1"},
        ],
    }
    datasets_json_value = _resolve_datasets_json_arg(args.datasets_json)
    if datasets_json_value:
        python_package_spec["env"].append({"name": "DATASETS_JSON", "value": datasets_json_value})
    if args.data_num_workers is not None:
        python_package_spec["env"].append({"name": "DATA_NUM_WORKERS", "value": str(args.data_num_workers)})

    machine_spec = {"machine_type": args.machine_type}
    if accelerator_count > 0:
        machine_spec["accelerator_type"] = args.accelerator_type
        machine_spec["accelerator_count"] = accelerator_count

    worker_pool_spec = {
        "replica_count": args.replica_count,
        "machine_spec": machine_spec,
        "python_package_spec": python_package_spec,
    }
    if args.boot_disk_size:
        worker_pool_spec["disk_spec"] = {
            "boot_disk_type": "pd-ssd",
            "boot_disk_size_gb": args.boot_disk_size,
        }

    job = aiplatform.CustomJob(
        display_name=display_name,
        worker_pool_specs=[worker_pool_spec],
        base_output_dir=base_output_dir,
    )

    job.run(sync=not args.async_run)


if __name__ == "__main__":
    main()
