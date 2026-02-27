import argparse
import subprocess
import os
import warnings

# Suppress the FutureWarnings concerning Python 3.10 End of Life 
# from the Google Cloud SDK, to keep stdout clean.
warnings.filterwarnings("ignore", category=FutureWarning, module="google.*")

from google.cloud import aiplatform
from google.cloud import storage

def build_and_upload_package(bucket_uri, project_id):
    """Builds a .tar.gz source distribution and uploads it to GCS natively"""
    print("1. Bundling your entire directory into a Python package...")
    
    # This command creates dist/distilled_llm-0.1.tar.gz
    subprocess.check_call(["python3", "setup.py", "sdist", "--formats=gztar"])
    
    # Find the newly created tar file
    dist_dir = "dist"
    files = os.listdir(dist_dir)
    package_file = [f for f in files if f.endswith('.tar.gz')][0]
    local_path = os.path.join(dist_dir, package_file)
    
    print(f"2. Uploading {package_file} to Cloud Storage...")
    storage_client = storage.Client(project=project_id)
    # bucket_uri looks like: gs://my-bucket
    bucket_name = bucket_uri.replace("gs://", "").split("/")[0]
    bucket = storage_client.bucket(bucket_name)
    
    # Upload path: gs://my-bucket/packages/distilled_llm-0.1.tar.gz
    gcs_blob_path = f"packages/{package_file}"
    blob = bucket.blob(gcs_blob_path)
    blob.upload_from_filename(local_path)
    
    package_gcs_uri = f"gs://{bucket_name}/{gcs_blob_path}"
    print(f"   Uploaded perfectly to: {package_gcs_uri}")
    
    return package_gcs_uri

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project_id", type=str, required=True, help="GCP Project ID")
    parser.add_argument("--region", type=str, required=True, help="GCP Region (e.g., us-central1)")
    parser.add_argument("--bucket_uri", type=str, required=True, help="GCS Bucket URI (e.g., gs://my-bucket)")
    args = parser.parse_args()

    # Automatically build and upload the package
    package_gcs_uri = build_and_upload_package(args.bucket_uri, args.project_id)

    aiplatform.init(project=args.project_id, location=args.region, staging_bucket=args.bucket_uri)

    MACHINE_TYPE = "g2-standard-24" # Features 2 NVIDIA L4 GPUs
    ACCELERATOR_TYPE = "NVIDIA_L4"
    ACCELERATOR_COUNT = 2
    
    # Upgraded from pytorch 2.2 to pytorch 2.4 to meet transformers>=5.2.0 dependencies
    TRAIN_IMAGE = "us-docker.pkg.dev/vertex-ai/training/pytorch-gpu.2-4.py310:latest"

    bucket_path = args.bucket_uri.replace("gs://", "").rstrip("/")
    output_dir_container = f"/gcs/{bucket_path}/test_vertex_outputs"

    print("3. Submitting Vertex AI Custom Job...")
    job = aiplatform.CustomJob(
        display_name="test-ddp-2xL4",
        worker_pool_specs=[
            {
                "replica_count": 1,
                "machine_spec": {
                    "machine_type": MACHINE_TYPE,
                    "accelerator_type": ACCELERATOR_TYPE,
                    "accelerator_count": ACCELERATOR_COUNT,
                },
                "python_package_spec": {
                    "executor_image_uri": TRAIN_IMAGE,
                    
                    # 1. Provide the exact path to the package we just built and uploaded
                    "package_uris": [package_gcs_uri], 
                    
                    # 2. Command to run: torchrun (No extra python code required!)
                    "python_module": "torch.distributed.run",
                    
                    # 3. Pass the torchrun arguments.
                    # Notice we can now use Python Module syntax: `--module text_vertex` or `--module scripts.trainer` 
                    "args": [
                        "--nproc_per_node=2",
                        "--module", "text_vertex", 
                        "--output_dir", output_dir_container
                    ],
                },
            }
        ],
        base_output_dir=f"{args.bucket_uri}/training_logs",
    )
    
    # Hang and stream the logs
    job.run(sync=True)
    
if __name__ == "__main__":
    main()
