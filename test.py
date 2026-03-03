import os
from huggingface_hub import hf_hub_download

def download_model():
    repo_id = "unsloth/Qwen3.5-9B-GGUF"
    filename = "Qwen3.5-9B-Q4_K_M.gguf"
    local_dir = "/media/prasanna/716F26140AED9B67/"
    
    # Ensure the directory exists
    os.makedirs(local_dir, exist_ok=True)
    
    print(f"Starting download of {filename} from {repo_id}...")
    print(f"Target directory: {local_dir}")
    
    # Use hf_hub_download with local_dir to avoid the standard cache structure
    # local_dir_use_symlinks=False ensures the file is physically present in the local_dir
    path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        local_dir=local_dir,
        local_dir_use_symlinks=False,
        resume_download=True  # Support resuming if interrupted
    )
    
    print(f"\nDownload complete! File located at: {path}")

if __name__ == "__main__":
    # If hf_transfer is installed, this will drastically speed up downloads
    # pip install hf_transfer - must be installed for this to work
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
    
    download_model()
