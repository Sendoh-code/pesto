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

cd "$REPO_DIR"

# ── prerequisite checks ───────────────────────────────────────────────────────
if ! command -v uv &>/dev/null; then
    echo "[setup] Error: 'uv' not found. Install it with:" >&2
    echo "  curl -LsSf https://astral.sh/uv/install.sh | sh" >&2
    exit 1
fi

if [ ! -f "$LMCACHE_SRC/pyproject.toml" ]; then
    echo "[setup] LMCache submodule is empty. Initialising..." >&2
    git -C "$REPO_DIR" submodule update --init --recursive
fi

# ── Python packages ───────────────────────────────────────────────────────────
echo "[setup] Installing uv-managed packages (vllm, boto3, ...)..."
uv sync

echo "[setup] Installing LMCache from source: $LMCACHE_SRC"
SETUPTOOLS_SCM_PRETEND_VERSION_FOR_LMCACHE="0.0.0" \
  uv pip install -e "$LMCACHE_SRC" --no-build-isolation

# ── MinIO binary ──────────────────────────────────────────────────────────────
echo "[setup] Checking for MinIO binary..."
if ! command -v minio &>/dev/null; then
    ARCH="$(uname -m)"
    case "$ARCH" in
        x86_64)  MINIO_ARCH="amd64" ;;
        aarch64) MINIO_ARCH="arm64" ;;
        *)
            echo "[setup] Warning: unsupported architecture '$ARCH' for auto-download." >&2
            echo "[setup] Install minio manually from https://min.io/download and put it on PATH." >&2
            MINIO_ARCH=""
            ;;
    esac

    if [ -n "$MINIO_ARCH" ]; then
        echo "[setup] minio not found on PATH — downloading linux-${MINIO_ARCH} to $REPO_DIR/bin/minio"
        mkdir -p "$REPO_DIR/bin"
        curl -fsSL "https://dl.min.io/server/minio/release/linux-${MINIO_ARCH}/minio" \
            -o "$REPO_DIR/bin/minio"
        chmod +x "$REPO_DIR/bin/minio"
        echo "[setup] MinIO downloaded: $REPO_DIR/bin/minio"
    fi
else
    echo "[setup] minio already on PATH: $(command -v minio)"
fi

echo "[setup] Done. Venv: $REPO_DIR/.venv"
echo "[setup] LMCache installed from: $LMCACHE_SRC"
