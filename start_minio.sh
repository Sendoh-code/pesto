#!/bin/bash
# MinIO object storage — API :9000 / Console :9001
# Data stored in ./minio-data/ (local to this repo)
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

MINIO_BIN="${REPO_DIR}/bin/minio"
if ! command -v minio &>/dev/null && [ ! -x "$MINIO_BIN" ]; then
    echo "Error: minio not found. Run ./setup.sh first." >&2
    exit 1
fi
MINIO_CMD=$(command -v minio 2>/dev/null || echo "$MINIO_BIN")

MINIO_ROOT_USER=minioadmin \
MINIO_ROOT_PASSWORD=minioadmin \
MINIO_DOMAIN=localhost \
"$MINIO_CMD" server "$REPO_DIR/minio-data" \
  --console-address :9001
