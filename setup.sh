#!/bin/bash
# One-time environment setup for pesto.
#
# Usage:
#   ./setup.sh                              # uses default LMCache source
#   LMCACHE_SRC=/path/to/LMCache ./setup.sh  # custom LMCache path
#
# After this script, start services with:
#   ./start_minio.sh && ./setup_bucket.sh   # first time only
#   ./start_controller.sh
#   ./start_instance0.sh
#   ./start_instance1.sh

set -euo pipefail
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LMCACHE_SRC="${LMCACHE_SRC:-$REPO_DIR/LMCache}"

echo "[setup] Creating uv venv..."
cd "$REPO_DIR"
uv venv

echo "[setup] Installing uv-managed packages (vllm, boto3, ...)..."
uv sync

echo "[setup] Installing LMCache from source: $LMCACHE_SRC"
uv pip install -e "$LMCACHE_SRC" --no-build-isolation

echo "[setup] Done. Venv: $REPO_DIR/.venv"
echo "[setup] LMCache installed from: $LMCACHE_SRC"
