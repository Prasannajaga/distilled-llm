import os
import sys
import logging
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

# Statically define training configuration for the Vertex AI Custom Job
STATIC_CONFIG = TrainingConfig(
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
    total_steps=5_000,
    warmup_steps=2000,
    
    # Checkpointing config
    ckpt_dir=os.environ.get("LOCAL_CKPT_DIR", "/outputs/mini-code-v1"),
    ckpt_interval_steps=5000,
    save_optimizer_state=True,
    
    # Logging
    enable_logging=True,
    log_interval_steps=100
)

# Vertex AI environment variables
AIP_MODEL_DIR = os.environ.get("AIP_MODEL_DIR", None)
AIP_CHECKPOINT_DIR = os.environ.get("AIP_CHECKPOINT_DIR", None)

def parse_gcs_uri(uri: str):
    if not uri or not uri.startswith("gs://"):
        return None, uri
    parts = uri[5:].split('/', 1)
    bucket_name = parts[0]
    prefix = parts[1] if len(parts) > 1 else ""
    return bucket_name, prefix

def save_checkpoint_bucket(local_path: str, bucket_name: str, gcs_path: str):
    """
    Uploads a local file to a GCS bucket.
    """
    try:
        logging.info(f"Uploading {local_path} to gs://{bucket_name}/{gcs_path}")
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(gcs_path)
        blob.upload_from_filename(local_path)
        logging.info(f"Successfully uploaded to gs://{bucket_name}/{gcs_path}")
    except Exception as e:
        logging.error(f"Failed to upload checkpoint to GCS: {e}")


def setup_distributed():
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size <= 1:
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

    return True, rank, world_size, local_rank


def cleanup_distributed():
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def patch_trainer_checkpointing(trainer_instance, gcs_checkpoint_dir: str):
    # Verify if it's a valid GCS path
    bucket_name, prefix = parse_gcs_uri(gcs_checkpoint_dir)
    if not bucket_name:
        logging.info("AIP_CHECKPOINT_DIR is not a GCS bucket. Checkpoints will only be saved locally.")
        return

    original_save_checkpoint = trainer_instance.save_checkpoint
    
    def patched_save_checkpoint(*args, **kwargs):
        # Save locally
        local_ckpt_path = original_save_checkpoint(*args, **kwargs)
        
        # Upload to bucket
        file_name = os.path.basename(local_ckpt_path)
        gcs_blob_path = os.path.join(prefix, file_name)
        
        save_checkpoint_bucket(local_ckpt_path, bucket_name, gcs_blob_path)
        return local_ckpt_path
        
    trainer_instance.save_checkpoint = patched_save_checkpoint


def build_dataloaders(
    bin_path: str,
    rank: int,
    world_size: int,
    is_distributed: bool,
):
    train_loader, val_loader = PackedDatasetBuilder.to_dataloader(
        bin_path=bin_path,
        block_size=STATIC_CONFIG.block_size,
        batch_size=STATIC_CONFIG.train_batch_size,
        val_ratio=0.20,              # 80/20 split
        val_batch_size=STATIC_CONFIG.eval_batch_size,
        num_workers=STATIC_CONFIG.num_workers,
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
        batch_size=STATIC_CONFIG.train_batch_size,
        sampler=train_sampler,
        num_workers=STATIC_CONFIG.num_workers,
        drop_last=True,
        pin_memory=True,
    )

    return train_loader, val_loader


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    is_distributed = False
    rank = 0
    world_size = 1
    local_rank = 0

    try:
        is_distributed, rank, world_size, local_rank = setup_distributed()
        is_main = rank == 0

        if not is_main:
            logging.getLogger().setLevel(logging.WARNING)

        logging.info(
            "Starting Vertex AI Training Job | rank=%s local_rank=%s world_size=%s",
            rank,
            local_rank,
            world_size,
        )

        if torch.cuda.is_available():
            device = torch.device(f"cuda:{local_rank}") if is_distributed else get_device(STATIC_CONFIG.device)
        else:
            device = get_device(STATIC_CONFIG.device)
        logging.info(f"Using device: {device}")
        
        # 1. Dataset Packing and Tokenization
        dataset_name = "ajibawa-2023/JavaScript-Code-Large"
        logging.info(f"Preparing Dataset: {dataset_name} (subset: 2 files only)")
        
        local_data_dir = "/tmp/data"
        os.makedirs(local_data_dir, exist_ok=True)
        
        tokenizer = load_tokenizer(TINYLLAMA_MODEL_NAME)
            
        # We pass data_files via load_kwargs to ensure we only download two files instead of the entire massive dataset
        builder = PackedDatasetBuilder(
            dataset_name=dataset_name,
            tokenizer=tokenizer,
            block_size=STATIC_CONFIG.block_size,
            output_path=local_data_dir,
            split="train",
            text_column="code",
            num_workers=STATIC_CONFIG.num_workers,
            batch_size=1024,
            data_files=[
                "java_script_only_0000.jsonl",
                "java_script_only_0001.jsonl"
            ]
        )
        
        # Build once per node (local_rank=0), then sync all ranks.
        if (not is_distributed) or local_rank == 0:
            builder.build()
        if is_distributed:
            dist.barrier()
        
        # 2. Setup Dataloaders with 80/20 train/val split
        logging.info("Setting up DataLoaders with 80/20 split")
        bin_path = os.path.join(local_data_dir, "data.bin")
        train_loader, val_loader = build_dataloaders(
            bin_path=bin_path,
            rank=rank,
            world_size=world_size,
            is_distributed=is_distributed,
        )
        
        # 3. Initialize Model
        logging.info("Initializing GQATransformer Model")
        model = GQATransformer(
            num_layers=STATIC_CONFIG.n_layer,
            n_emb=STATIC_CONFIG.n_embd,
            n_head=STATIC_CONFIG.n_head,
            n_kv_head=STATIC_CONFIG.n_head // 2,
            vocab_size=len(tokenizer),
            block_size=STATIC_CONFIG.block_size,
        )
        
        # 4. Initialize Production-Grade Trainer
        logging.info("Initializing Trainer")
        trainer = Trainer(
            model=model,
            config=STATIC_CONFIG,
            device=device,
            tokenizer=tokenizer,
            ckpt_dir=STATIC_CONFIG.ckpt_dir,
        )
        
        # Patch trainer checkpointing only on rank 0.
        if is_main and AIP_CHECKPOINT_DIR:
            logging.info(f"Patching checkpointer to upload to: {AIP_CHECKPOINT_DIR}")
            patch_trainer_checkpointing(trainer, AIP_CHECKPOINT_DIR)
            
        # 5. Execute Training Loop
        logging.info("Starting Training Loop")
        if is_distributed:
            trainer.train_distributed(
                train_dataloader=train_loader,
                val_dataloader=val_loader,
                epochs=1,
                max_steps=STATIC_CONFIG.total_steps
            )
        else:
            trainer.train(
                train_dataloader=train_loader,
                val_dataloader=val_loader,
                epochs=1,
                max_steps=STATIC_CONFIG.total_steps
            )
        
        # Final upload of the model if AIP_MODEL_DIR is specified (main process only)
        if is_main and AIP_MODEL_DIR:
            logging.info(f"Training complete. Uploading final model to {AIP_MODEL_DIR}")
            bucket_name, prefix = parse_gcs_uri(AIP_MODEL_DIR)
            if bucket_name:
                final_ckpt = os.path.join(STATIC_CONFIG.ckpt_dir, f"final_ckpt_{trainer.global_step}.pt")
                if os.path.exists(final_ckpt):
                    file_name = os.path.basename(final_ckpt)
                    gcs_blob_path = os.path.join(prefix, file_name)
                    save_checkpoint_bucket(final_ckpt, bucket_name, gcs_blob_path)
                else:
                    logging.warning(f"Final checkpoint not found at {final_ckpt}")
    finally:
        cleanup_distributed()

if __name__ == "__main__":
    main()
