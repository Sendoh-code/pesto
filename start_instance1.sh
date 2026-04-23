#!/bin/bash
# vLLM instance 1 — port 8001 / lmcache worker port 8051
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$REPO_DIR/.venv/bin/activate"

unset LD_LIBRARY_PATH
export AWS_ACCESS_KEY_ID="minioadmin"
export AWS_SECRET_ACCESS_KEY="minioadmin"
export AWS_ENDPOINT_URL="http://localhost:9000"
export AWS_DEFAULT_REGION="us-east-1"
export PYTHONHASHSEED=0
export LD_LIBRARY_PATH=$(python -c "import torch, os; print(os.path.dirname(torch.__file__))")/lib
export LMCACHE_CONFIG_FILE="$REPO_DIR/lmcache_instance1.yaml"

python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen2.5-0.5B \
  --port 8001 \
  --gpu-memory-utilization 0.4 \
  --enable-chunked-prefill \
  --kv-transfer-config "{\"kv_connector\":\"LMCacheConnectorV1\",\"kv_role\":\"kv_both\",\"kv_connector_extra_config\":{\"lmcache_config_file\":\"$REPO_DIR/lmcache_instance1.yaml\"}}"
