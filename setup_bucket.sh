#!/bin/bash
# Create the MinIO bucket used by LMCache. Run once after start_minio.sh.
set -euo pipefail
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$REPO_DIR/.venv/bin/activate"

python3 - <<'EOF'
import boto3, botocore

s3 = boto3.client(
    "s3",
    endpoint_url="http://localhost:9000",
    aws_access_key_id="minioadmin",
    aws_secret_access_key="minioadmin",
    region_name="us-east-1",
    config=botocore.config.Config(s3={"addressing_style": "path"}),
)
try:
    s3.create_bucket(Bucket="lmcache-bucket")
    print("Bucket 'lmcache-bucket' created.")
except s3.exceptions.BucketAlreadyOwnedByYou:
    print("Bucket 'lmcache-bucket' already exists.")
except Exception as e:
    print(f"Error: {e}")
EOF
