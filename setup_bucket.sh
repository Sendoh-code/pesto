#!/bin/bash
# Create the MinIO bucket used by LMCache. Run once after start_minio.sh.
set -euo pipefail
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$REPO_DIR/.venv/bin/activate"

echo "[bucket] Waiting for MinIO to be ready..."
for i in $(seq 1 30); do
    if curl -sf "http://localhost:9000/minio/health/live" &>/dev/null; then
        break
    fi
    if [ "$i" -eq 30 ]; then
        echo "[bucket] Error: MinIO did not become ready after 30 seconds." >&2
        echo "[bucket] Make sure start_minio.sh is running." >&2
        exit 1
    fi
    sleep 1
done
echo "[bucket] MinIO is ready."

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
