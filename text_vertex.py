import os
import argparse
import logging
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler

# ---------------------------------------------------------------------------
# Setup Logger
# We set up a precise logger to see timestamps and severities in Vertex AI
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - RANK [%(rank)s] - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Provide a default rank=0 for the root logger before DDP is setup
old_factory = logging.getLogRecordFactory()
def record_factory(*args, **kwargs):
    record = old_factory(*args, **kwargs)
    record.rank = os.environ.get("LOCAL_RANK", "None")
    return record
logging.setLogRecordFactory(record_factory)


class SimpleDataset(Dataset):
    def __init__(self):
        # A simple array of length 10
        self.data = torch.arange(10, dtype=torch.float32).unsqueeze(1)
        self.target = self.data * 2.0  # f(x) = 2x

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.target[idx]

class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self, x):
        return self.linear(x)

def setup():
    # Initializes the distributed environment using nccl for GPUs
    dist.init_process_group("nccl")

def cleanup():
    dist.destroy_process_group()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, required=True, 
                        help="Directory to save the model. Vertex AI mounts GCS bucket directly here.")
    args = parser.parse_args()

    # 1. Setup Distributed
    setup()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    # Control verbosity: Only Rank 0 prints INFO level messages to prevent log spam.
    # Other ranks only print WARNING or higher.
    if local_rank == 0:
        logger.setLevel(logging.INFO)
    else:
        logger.setLevel(logging.WARNING)

    logger.info("Distributed process group initialized perfectly.")
    logger.info(f"Arguments parsed: {args}")

    # 2. Setup Dataset and Sampler
    dataset = SimpleDataset()
    sampler = DistributedSampler(dataset)
    dataloader = DataLoader(dataset, batch_size=2, sampler=sampler, shuffle=False)
    logger.info(f"Dataloader and Sampler setup complete. Total samples: {len(dataset)}")

    # 3. Setup Model and DDP
    model = SimpleModel().to(device)
    model = DDP(model, device_ids=[local_rank])
    logger.info("Model transferred to GPU and wrapped in DDP successfully.")

    optimizer = optim.SGD(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()

    # 4. Training Loop
    epochs = 5
    logger.info("Starting training loop...")
    
    model.train()
    for epoch in range(epochs):
        sampler.set_epoch(epoch)
        for batch_idx, (x, y) in enumerate(dataloader):
            x, y = x.to(device), y.to(device)
            
            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output, y)
            
            loss.backward()  # Gradients sync across GPUs here
            optimizer.step()
            
            # Print only from rank 0 to avoid duplicate outputs
            if local_rank == 0:
                logger.info(f"Epoch {epoch} | Batch {batch_idx} | Loss: {loss.item():.4f}")

    # 5. Save Model (Only Rank 0 saves!)
    if local_rank == 0:
        os.makedirs(args.output_dir, exist_ok=True)
        save_path = os.path.join(args.output_dir, "test_vertex_model.pt")
        logger.info(f"Saving model to {save_path}...")
        
        # Save from model.module to drop the DDP wrapper
        torch.save(model.module.state_dict(), save_path)
        logger.info("Model safely saved to GCS bucket!")

    # 6. Clean up
    logger.info("Destroying process group and cleaning up...")
    cleanup()

if __name__ == "__main__":
    main()
