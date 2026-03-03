import argparse
import copy
import json
import os
import sys
import logging
import faulthandler
import io
import time
from contextlib import contextmanager, redirect_stdout, redirect_stderr
from google.cloud import storage
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

# Ensure we can import from the rest of the workspace
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.config import TrainingConfig
from utils.trainer import Trainer
from utils.packed_dataset_builder import PackedDatasetBuilder
from utils.common import get_device
from scripts.model import GQATransformer
from Cdatasets.tokenizer import load_tokenizer

TINYLLAMA_MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0" 
LOGGER = logging.getLogger("train_vertex")

BASE_CONFIG = TrainingConfig(
    device="cuda" if torch.cuda.is_available() else "cpu",
    seed=42,
    num_workers=int(os.cpu_count() or 4),
    
    # Model parameters
    n_layer=12,
    n_embd=768,
    n_head=12,
    block_size=512,
    
    # Data parameters
    train_batch_size=16,
    eval_batch_size=8,
    grad_accum_steps=4,
    validation_batch_count=20,
    
    # Optimization
    optimizer="adamw",
    lr=5e-4,
    weight_decay=0.1,
    grad_clip_norm=1.0,
    use_amp=True,
    amp_dtype="bfloat16" if torch.cuda.is_bf16_supported() else "float16",
    
    # LR Schedule
    total_steps=1000,
    warmup_steps=100,
    
    # Checkpointing config
    ckpt_dir=os.environ.get("LOCAL_CKPT_DIR", "/outputs/mini-code-v1"),
    ckpt_interval_steps=500,
    save_optimizer_state=True,
    
    # Logging
    enable_logging=True,
    log_interval_steps=100
)

# Vertex AI environment variables
AIP_MODEL_DIR = os.environ.get("AIP_MODEL_DIR", None)
AIP_CHECKPOINT_DIR = os.environ.get("AIP_CHECKPOINT_DIR", None)


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    return int(value) if value is not None else int(default)


def _env_float(name: str, default: float) -> float:
    value = os.environ.get(name)
    return float(value) if value is not None else float(default)


def _env_str(name: str, default: str) -> str:
    value = os.environ.get(name)
    return value if value is not None else str(default)


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return bool(default)
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train model on one packed binary file."
    )
    parser.add_argument(
        "--bin_path",
        type=str,
        default=os.environ.get("DATA_BIN_PATH", "/tmp/vertex-bins/data.bin"),
        help="Path to packed binary (.bin) produced by scripts.loadDataset.",
    )
    parser.add_argument("--device", type=str, default=_env_str("TRAIN_DEVICE", BASE_CONFIG.device))
    parser.add_argument("--seed", type=int, default=_env_int("TRAIN_SEED", BASE_CONFIG.seed or 42))
    parser.add_argument("--num_workers", type=int, default=_env_int("TRAIN_NUM_WORKERS", BASE_CONFIG.num_workers))

    parser.add_argument("--n_layer", type=int, default=_env_int("TRAIN_N_LAYER", BASE_CONFIG.n_layer))
    parser.add_argument("--n_embd", type=int, default=_env_int("TRAIN_N_EMBD", BASE_CONFIG.n_embd))
    parser.add_argument("--n_head", type=int, default=_env_int("TRAIN_N_HEAD", BASE_CONFIG.n_head))
    parser.add_argument(
        "--block_size",
        type=int,
        default=_env_int("TRAIN_BLOCK_SIZE", _env_int("BLOCK_SIZE", BASE_CONFIG.block_size)),
    )

    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=_env_int("TRAIN_TRAIN_BATCH_SIZE", BASE_CONFIG.train_batch_size),
    )
    parser.add_argument(
        "--eval_batch_size",
        type=int,
        default=_env_int("TRAIN_EVAL_BATCH_SIZE", BASE_CONFIG.eval_batch_size),
    )
    parser.add_argument(
        "--grad_accum_steps",
        type=int,
        default=_env_int("TRAIN_GRAD_ACCUM_STEPS", BASE_CONFIG.grad_accum_steps),
    )
    parser.add_argument(
        "--validation_batch_count",
        type=int,
        default=_env_int("TRAIN_VALIDATION_BATCH_COUNT", BASE_CONFIG.validation_batch_count),
    )
    parser.add_argument("--val_ratio", type=float, default=_env_float("TRAIN_VAL_RATIO", 0.20))

    parser.add_argument("--optimizer", type=str, default=_env_str("TRAIN_OPTIMIZER", BASE_CONFIG.optimizer))
    parser.add_argument("--lr", type=float, default=_env_float("TRAIN_LR", BASE_CONFIG.lr))
    parser.add_argument("--eps", type=float, default=_env_float("TRAIN_EPS", BASE_CONFIG.eps))
    parser.add_argument("--weight_decay", type=float, default=_env_float("TRAIN_WEIGHT_DECAY", BASE_CONFIG.weight_decay))
    parser.add_argument("--grad_clip_norm", type=float, default=_env_float("TRAIN_GRAD_CLIP_NORM", BASE_CONFIG.grad_clip_norm))
    parser.add_argument("--use_amp", type=int, choices=[0, 1], default=1 if _env_bool("TRAIN_USE_AMP", BASE_CONFIG.use_amp) else 0)
    parser.add_argument("--amp_dtype", type=str, default=_env_str("TRAIN_AMP_DTYPE", BASE_CONFIG.amp_dtype))
    parser.add_argument(
        "--enable_gradient_checkpointing",
        type=int,
        choices=[0, 1],
        default=1 if _env_bool("TRAIN_ENABLE_GRADIENT_CHECKPOINTING", BASE_CONFIG.enable_gradient_checkpointing) else 0,
    )

    parser.add_argument("--total_steps", type=int, default=_env_int("TRAIN_TOTAL_STEPS", BASE_CONFIG.total_steps))
    parser.add_argument("--warmup_steps", type=int, default=_env_int("TRAIN_WARMUP_STEPS", BASE_CONFIG.warmup_steps))

    parser.add_argument(
        "--ckpt_dir",
        type=str,
        default=os.environ.get("LOCAL_CKPT_DIR", _env_str("TRAIN_CKPT_DIR", BASE_CONFIG.ckpt_dir)),
    )
    parser.add_argument(
        "--ckpt_interval_steps",
        type=int,
        default=_env_int("TRAIN_CKPT_INTERVAL_STEPS", BASE_CONFIG.ckpt_interval_steps),
    )
    parser.add_argument(
        "--save_optimizer_state",
        type=int,
        choices=[0, 1],
        default=1 if _env_bool("TRAIN_SAVE_OPTIMIZER_STATE", BASE_CONFIG.save_optimizer_state) else 0,
    )
    parser.add_argument(
        "--log_interval_steps",
        type=int,
        default=_env_int("TRAIN_LOG_INTERVAL_STEPS", BASE_CONFIG.log_interval_steps),
    )
    return parser.parse_args()


def build_training_config(args: argparse.Namespace) -> TrainingConfig:
    cfg = copy.deepcopy(BASE_CONFIG)
    cfg.device = args.device
    cfg.seed = int(args.seed)
    cfg.num_workers = int(args.num_workers)

    cfg.n_layer = int(args.n_layer)
    cfg.n_embd = int(args.n_embd)
    cfg.n_head = int(args.n_head)
    cfg.block_size = int(args.block_size)

    cfg.train_batch_size = int(args.train_batch_size)
    cfg.eval_batch_size = int(args.eval_batch_size)
    cfg.grad_accum_steps = int(args.grad_accum_steps)
    cfg.validation_batch_count = int(args.validation_batch_count)

    cfg.optimizer = args.optimizer
    cfg.lr = float(args.lr)
    cfg.eps = float(args.eps)
    cfg.weight_decay = float(args.weight_decay)
    cfg.grad_clip_norm = float(args.grad_clip_norm)
    cfg.use_amp = bool(args.use_amp)
    cfg.amp_dtype = args.amp_dtype
    cfg.enable_gradient_checkpointing = bool(args.enable_gradient_checkpointing)

    cfg.total_steps = int(args.total_steps)
    cfg.warmup_steps = int(args.warmup_steps)

    cfg.ckpt_dir = args.ckpt_dir
    cfg.ckpt_interval_steps = int(args.ckpt_interval_steps)
    cfg.save_optimizer_state = bool(args.save_optimizer_state)
    cfg.log_interval_steps = int(args.log_interval_steps)
    return cfg


class RankContextFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.rank = os.environ.get("RANK", "0")
        record.local_rank = os.environ.get("LOCAL_RANK", "0")
        record.world_size = os.environ.get("WORLD_SIZE", "1")
        return True


def _fmt_kv(**kwargs):
    parts = []
    for key, value in kwargs.items():
        if value is None:
            continue
        parts.append(f"{key}={value}")
    return " ".join(parts)


@contextmanager
def log_stage(name: str, **metadata):
    started = time.perf_counter()
    meta = _fmt_kv(**metadata)
    if meta:
        LOGGER.info("stage=start name=%s %s", name, meta)
    else:
        LOGGER.info("stage=start name=%s", name)
    try:
        yield
    except Exception:
        if meta:
            LOGGER.exception("stage=failed name=%s %s", name, meta)
        else:
            LOGGER.exception("stage=failed name=%s", name)
        raise
    else:
        elapsed = time.perf_counter() - started
        if meta:
            LOGGER.info("stage=done name=%s elapsed_s=%.2f %s", name, elapsed, meta)
        else:
            LOGGER.info("stage=done name=%s elapsed_s=%.2f", name, elapsed)


class _StreamToLogger(io.TextIOBase):
    def __init__(self, level: int = logging.INFO, prefix: str = ""):
        super().__init__()
        self.level = level
        self.prefix = prefix
        self._buf = ""

    def write(self, data: str) -> int:
        if not data:
            return 0
        self._buf += data
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            line = line.strip()
            if line:
                LOGGER.log(self.level, "%s%s", self.prefix, line)
        return len(data)

    def flush(self) -> None:
        line = self._buf.strip()
        if line:
            LOGGER.log(self.level, "%s%s", self.prefix, line)
        self._buf = ""


@contextmanager
def capture_prints(level: int = logging.INFO, prefix: str = ""):
    stream = _StreamToLogger(level=level, prefix=prefix)
    with redirect_stdout(stream), redirect_stderr(stream):
        yield
    stream.flush()


def configure_logging():
    log_level = os.environ.get("TRAIN_LOG_LEVEL", "INFO").upper()
    root_logger = logging.getLogger()
    root_logger.handlers.clear()

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s | %(levelname)s | r%(rank)s/l%(local_rank)s/w%(world_size)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    handler.addFilter(RankContextFilter())
    root_logger.addHandler(handler)
    root_logger.setLevel(getattr(logging, log_level, logging.INFO))

    for noisy_logger in ("urllib3", "google", "google.auth", "google.api_core", "datasets", "transformers"):
        logging.getLogger(noisy_logger).setLevel(logging.WARNING)

    try:
        faulthandler.enable(all_threads=True)
    except Exception:
        LOGGER.warning("Failed to enable faulthandler")


def parse_gcs_uri(uri: str):
    if not uri or not uri.startswith("gs://"):
        return None, uri
    parts = uri[5:].split('/', 1)
    bucket_name = parts[0]
    prefix = parts[1] if len(parts) > 1 else ""
    return bucket_name, prefix


def safe_len(value) -> int | None:
    try:
        return int(len(value))
    except Exception:
        return None


def load_dataset_build_summary(bin_path: str) -> dict | None:
    bin_dir = os.path.dirname(os.path.abspath(bin_path))
    summary_path = os.path.join(bin_dir, "dataset_build_summary.json")
    if not os.path.exists(summary_path):
        return None
    try:
        with open(summary_path, "r", encoding="utf-8") as fp:
            summary = json.load(fp)
        if isinstance(summary, dict):
            summary["summary_path"] = summary_path
            return summary
    except Exception:
        LOGGER.exception("Failed to load dataset build summary from %s", summary_path)
    return None

def save_checkpoint_bucket(local_path: str, bucket_name: str, gcs_path: str):
    """
    Uploads a local file to a GCS bucket.
    """
    try:
        LOGGER.info("gcs_upload=start local=%s target=gs://%s/%s", local_path, bucket_name, gcs_path)
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(gcs_path)
        blob.upload_from_filename(local_path)
        LOGGER.info("gcs_upload=done target=gs://%s/%s", bucket_name, gcs_path)
    except Exception:
        LOGGER.exception(
            "Failed to upload checkpoint to GCS | local_path=%s | bucket=%s | gcs_path=%s",
            local_path,
            bucket_name,
            gcs_path,
        )


def upload_auxiliary_artifacts(local_ckpt_dir: str, bucket_name: str, prefix: str, uploaded: set[str]):
    if not os.path.isdir(local_ckpt_dir):
        return
    allowlist = {
        "config.json",
        "results.json",
        "special_tokens_map.json",
        "added_tokens.json",
        "vocab.json",
        "merges.txt",
        "spiece.model",
        "sentencepiece.bpe.model",
    }
    for fname in os.listdir(local_ckpt_dir):
        local_path = os.path.join(local_ckpt_dir, fname)
        if not os.path.isfile(local_path):
            continue
        if fname in uploaded:
            continue
        if fname in allowlist or fname.startswith("tokenizer"):
            gcs_blob_path = os.path.join(prefix, fname)
            save_checkpoint_bucket(local_path, bucket_name, gcs_blob_path)
            uploaded.add(fname)


def upload_model_bundle(
    local_ckpt_dir: str,
    bucket_name: str,
    prefix: str,
    checkpoint_file: str,
    results_file: str | None = None,
):
    if os.path.exists(checkpoint_file):
        file_name = os.path.basename(checkpoint_file)
        save_checkpoint_bucket(checkpoint_file, bucket_name, os.path.join(prefix, file_name))
    else:
        LOGGER.warning("Checkpoint not found for bundle upload: %s", checkpoint_file)
    if results_file:
        if os.path.exists(results_file):
            results_name = os.path.basename(results_file)
            save_checkpoint_bucket(results_file, bucket_name, os.path.join(prefix, results_name))
        else:
            LOGGER.warning("Results file not found for bundle upload: %s", results_file)
    upload_auxiliary_artifacts(local_ckpt_dir, bucket_name, prefix, uploaded=set())


def setup_distributed():
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size <= 1:
        LOGGER.info("distributed=disabled world_size=%s", world_size)
        return False, 0, 1, 0

    if not dist.is_available():
        raise RuntimeError("torch.distributed is not available but WORLD_SIZE > 1")

    backend = "nccl" if torch.cuda.is_available() else "gloo"
    if not dist.is_initialized():
        dist.init_process_group(backend=backend, init_method="env://")

    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = dist.get_world_size()

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)

    LOGGER.info(
        "distributed=initialized backend=%s rank=%s local_rank=%s world_size=%s",
        backend,
        rank,
        local_rank,
        world_size,
    )
    return True, rank, world_size, local_rank


def cleanup_distributed():
    try:
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()
            LOGGER.info("distributed=destroyed")
    except Exception:
        LOGGER.exception("Failed while destroying distributed process group")


def patch_trainer_checkpointing(trainer_instance, gcs_checkpoint_dir: str):
    # Verify if it's a valid GCS path
    bucket_name, prefix = parse_gcs_uri(gcs_checkpoint_dir)
    if not bucket_name:
        LOGGER.info("checkpoint_upload=disabled reason=non_gcs_dir value=%s", gcs_checkpoint_dir)
        return

    original_save_checkpoint = trainer_instance.save_checkpoint
    uploaded_aux_files: set[str] = set()
    
    def patched_save_checkpoint(*args, **kwargs):
        # Save locally
        local_ckpt_path = original_save_checkpoint(*args, **kwargs)
        
        # Upload to bucket
        file_name = os.path.basename(local_ckpt_path)
        gcs_blob_path = os.path.join(prefix, file_name)
        
        save_checkpoint_bucket(local_ckpt_path, bucket_name, gcs_blob_path)
        upload_auxiliary_artifacts(
            local_ckpt_dir=trainer_instance.ckpt_dir,
            bucket_name=bucket_name,
            prefix=prefix,
            uploaded=uploaded_aux_files,
        )
        return local_ckpt_path
        
    trainer_instance.save_checkpoint = patched_save_checkpoint


def build_dataloaders(
    bin_path: str,
    config: TrainingConfig,
    val_ratio: float,
    rank: int,
    world_size: int,
    is_distributed: bool,
):
    train_loader, val_loader = PackedDatasetBuilder.to_dataloader(
        bin_path=bin_path,
        block_size=config.block_size,
        batch_size=config.train_batch_size,
        val_ratio=val_ratio,
        val_batch_size=config.eval_batch_size,
        num_workers=config.num_workers,
        shuffle=not is_distributed,
        drop_last=True,
        pin_memory=True,
    )

    if not is_distributed:
        return train_loader, val_loader

    train_sampler = DistributedSampler(
        train_loader.dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        drop_last=True,
    )
    train_loader = DataLoader(
        train_loader.dataset,
        batch_size=config.train_batch_size,
        sampler=train_sampler,
        num_workers=config.num_workers,
        drop_last=True,
        pin_memory=True,
    )

    return train_loader, val_loader


def main():
    args = parse_args()
    config = build_training_config(args)
    configure_logging()
    is_distributed = False
    rank = 0
    world_size = 1
    local_rank = 0

    try:
        with log_stage("distributed_setup"):
            is_distributed, rank, world_size, local_rank = setup_distributed()

        is_main = rank == 0

        if not is_main:
            logging.getLogger().setLevel(logging.ERROR)

        LOGGER.info(
            "run=start rank=%s local_rank=%s world_size=%s",
            rank,
            local_rank,
            world_size,
        )
        LOGGER.info(
            "env %s",
            _fmt_kv(
                aip_model_dir=AIP_MODEL_DIR,
                aip_checkpoint_dir=AIP_CHECKPOINT_DIR,
                local_ckpt_dir=config.ckpt_dir,
                total_steps=config.total_steps,
                grad_accum_steps=config.grad_accum_steps,
                bin_path=args.bin_path,
                n_layer=config.n_layer,
                n_embd=config.n_embd,
                n_head=config.n_head,
                block_size=config.block_size,
                train_batch_size=config.train_batch_size,
                eval_batch_size=config.eval_batch_size,
                lr=config.lr,
            ),
        )

        if torch.cuda.is_available():
            device = torch.device(f"cuda:{local_rank}") if is_distributed else get_device(config.device)
        else:
            device = get_device(config.device)
        LOGGER.info("device=%s", device)

        bin_path = os.path.abspath(args.bin_path)
        LOGGER.info("dataset bin_path=%s", bin_path)

        with log_stage("tokenizer_load", model=TINYLLAMA_MODEL_NAME):
            tokenizer = load_tokenizer(TINYLLAMA_MODEL_NAME)

        with log_stage("dataloader_build", bin_path=bin_path):
            with capture_prints(level=logging.INFO, prefix="[data] "):
                train_loader, val_loader = build_dataloaders(
                    bin_path=bin_path,
                    config=config,
                    val_ratio=args.val_ratio,
                    rank=rank,
                    world_size=world_size,
                    is_distributed=is_distributed,
                )
            val_batches = len(val_loader) if val_loader is not None else 0
            LOGGER.info("dataloader %s", _fmt_kv(train_batches=len(train_loader), val_batches=val_batches))

        with log_stage("model_init", model="GQATransformer"):
            model = GQATransformer(
                num_layers=config.n_layer,
                n_emb=config.n_embd,
                n_head=config.n_head,
                n_kv_head=config.n_head // 2,
                vocab_size=len(tokenizer),
                block_size=config.block_size,
            )

        with log_stage("trainer_init"):
            trainer = Trainer(
                model=model,
                config=config,
                device=device,
                tokenizer=tokenizer,
                ckpt_dir=config.ckpt_dir,
            )

        if is_main and AIP_CHECKPOINT_DIR:
            with log_stage("checkpoint_patch", gcs_dir=AIP_CHECKPOINT_DIR):
                patch_trainer_checkpointing(trainer, AIP_CHECKPOINT_DIR)

        with log_stage("training", distributed=is_distributed, target_steps=config.total_steps):
            if is_distributed:
                trainer.train_distributed(
                    train_dataloader=train_loader,
                    val_dataloader=val_loader,
                    epochs=1,
                    max_steps=config.total_steps
                )
            else:
                trainer.train(
                    train_dataloader=train_loader,
                    val_dataloader=val_loader,
                    epochs=1,
                    max_steps=config.total_steps
                )

        final_ckpt = None
        results_json_path = None
        if is_main and trainer.ckpt_dir:
            with log_stage("final_checkpoint_save", step=trainer.global_step):
                final_ckpt = trainer.save_final_checkpoint(step=trainer.global_step)

        if is_main:
            with log_stage("results_save", ckpt_dir=config.ckpt_dir):
                train_samples = safe_len(getattr(train_loader, "dataset", None))
                val_samples = 0
                if val_loader is not None:
                    val_samples = safe_len(getattr(val_loader, "dataset", None)) or 0
                total_sequences = None
                if train_samples is not None:
                    total_sequences = int(train_samples) + int(val_samples)

                dataset_summary = load_dataset_build_summary(bin_path)
                summary_total_sequences = None
                summary_total_tokens = None
                summary_total_dataset_samples = None
                summary_dataset_row_counts = None
                summary_datasets = None
                summary_path = None
                if dataset_summary:
                    summary_total_sequences = dataset_summary.get("total_sequences")
                    summary_total_tokens = dataset_summary.get("total_tokens")
                    summary_total_dataset_samples = dataset_summary.get("total_dataset_samples")
                    summary_dataset_row_counts = dataset_summary.get("dataset_row_counts")
                    summary_datasets = dataset_summary.get("datasets")
                    summary_path = dataset_summary.get("summary_path")
                    if total_sequences is None and summary_total_sequences is not None:
                        total_sequences = int(summary_total_sequences)

                results_extra = {
                    "dataset": {
                        "bin_path": bin_path,
                        "dataset_build_summary_path": summary_path,
                        "source_datasets": summary_datasets,
                        "source_dataset_count": len(summary_datasets) if isinstance(summary_datasets, list) else None,
                        "source_dataset_rows": summary_dataset_row_counts,
                        "total_dataset_samples": summary_total_dataset_samples,
                        "train_split_samples": train_samples,
                        "val_split_samples": val_samples,
                        "total_sequences": total_sequences,
                        "packed_total_sequences": summary_total_sequences,
                        "packed_total_tokens": summary_total_tokens,
                    },
                    "vertex": {
                        "aip_model_dir": AIP_MODEL_DIR,
                        "aip_checkpoint_dir": AIP_CHECKPOINT_DIR,
                        "rank": rank,
                        "local_rank": local_rank,
                        "world_size": world_size,
                        "is_distributed": is_distributed,
                    },
                    "artifacts": {
                        "ckpt_dir": config.ckpt_dir,
                        "final_checkpoint_path": final_ckpt,
                        "step_metrics_path": trainer.step_metrics_path,
                    },
                }
                results_json_path = trainer.save_results_json(extra=results_extra)
                LOGGER.info("results=written path=%s", results_json_path)

        if is_main and AIP_MODEL_DIR:
            with log_stage("final_model_upload", target=AIP_MODEL_DIR):
                bucket_name, prefix = parse_gcs_uri(AIP_MODEL_DIR)
                if bucket_name:
                    if not final_ckpt:
                        final_ckpt = os.path.join(config.ckpt_dir, f"final_ckpt_{trainer.global_step}.pt")
                    if not os.path.exists(final_ckpt):
                        interval_ckpt = os.path.join(config.ckpt_dir, f"model_{trainer.global_step}.pt")
                        if os.path.exists(interval_ckpt):
                            final_ckpt = interval_ckpt
                    upload_model_bundle(
                        local_ckpt_dir=config.ckpt_dir,
                        bucket_name=bucket_name,
                        prefix=prefix,
                        checkpoint_file=final_ckpt,
                        results_file=results_json_path,
                    )
        elif is_main and AIP_CHECKPOINT_DIR and results_json_path:
            with log_stage("results_upload", target=AIP_CHECKPOINT_DIR):
                bucket_name, prefix = parse_gcs_uri(AIP_CHECKPOINT_DIR)
                if bucket_name:
                    gcs_blob_path = os.path.join(prefix, os.path.basename(results_json_path))
                    save_checkpoint_bucket(results_json_path, bucket_name, gcs_blob_path)

        LOGGER.info(
            "run=complete %s",
            _fmt_kv(
                global_step=trainer.global_step,
                best_train_loss=f"{trainer._best_train_loss:.4f}" if trainer._best_train_loss != float("inf") else None,
                best_val_loss=f"{trainer._best_val_loss:.4f}" if trainer._best_val_loss != float("inf") else None,
                results_json=results_json_path,
            ),
        )
    except Exception:
        LOGGER.exception("run=fatal")
        raise
    finally:
        cleanup_distributed()

if __name__ == "__main__":
    main()
