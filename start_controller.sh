#!/bin/bash
# LMCache controller — API :8100 / monitor :8200
# pull :8300  reply :8400
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$REPO_DIR/.venv/bin/activate"

python3 -m lmcache.v1.api_server --port 8100
