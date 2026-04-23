#!/bin/bash
# MinIO object storage — API :9000 / Console :9001
# Data stored in ./minio-data/ (local to this repo)
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

MINIO_ROOT_USER=minioadmin \
MINIO_ROOT_PASSWORD=minioadmin \
MINIO_DOMAIN=localhost \
minio server "$REPO_DIR/minio-data" \
  --console-address :9001
