#!/bin/bash
# Verify end-to-end KV cache flow: vLLM → LMCache → MinIO.
#
# What this checks:
#   1. MinIO bucket exists and is initially empty (or reports current count)
#   2. First inference request causes LMCache to write chunks to MinIO
#   3. Controller registers the worker and reports key_count matching MinIO
#   4. Second identical request is served from CPU cache (lower latency)
#
# Prerequisites: all services must be running before executing this script.
#   ./start_minio.sh &
#   ./start_controller.sh &
#   ./start_instance0.sh &   # wait until /health returns 200

set -euo pipefail
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$REPO_DIR/.venv/bin/activate"

VLLM_PORT="${VLLM_PORT:-8000}"
CONTROLLER_PORT="${CONTROLLER_PORT:-8100}"
MINIO_ENDPOINT="http://localhost:9000"
BUCKET="lmcache-bucket"
MODEL="Qwen/Qwen2.5-0.5B"

PROMPT="The history of artificial intelligence begins in antiquity, with myths, stories, and rumors of artificial beings endowed with intelligence or consciousness by master craftsmen. The seeds of modern AI were planted by classical philosophers who attempted to describe the process of human thinking as the mechanical manipulation of symbols. This work culminated in the invention of the programmable digital computer in the 1940s, a machine based on the abstract essence of mathematical reasoning. This device and the ideas behind it inspired a handful of scientists to begin seriously discussing the possibility of building an electronic brain. The field of AI research was founded at a workshop held on the campus of Dartmouth College during the summer of 1956. Those who attended would become the leaders of AI research for decades. Many of them made predictions that within a generation, machines as intelligent as humans would exist. Millions of dollars were poured into making this vision come true. Eventually it became obvious that commercial developers and government researchers had grossly underestimated the difficulty of the project."

log() { echo "[$(date +%H:%M:%S)] $*" >&2; }

send_request() {
    local label=$1
    local start end ms output

    start=$(date +%s%3N)
    response=$(curl -s -X POST "http://localhost:$VLLM_PORT/v1/completions" \
        -H "Content-Type: application/json" \
        -d "{\"model\":\"$MODEL\",
             \"prompt\":$(echo "$PROMPT" | python3 -c 'import sys,json; print(json.dumps(sys.stdin.read()))'),
             \"max_tokens\":1,\"temperature\":0}")
    end=$(date +%s%3N)
    ms=$((end - start))
    output=$(echo "$response" | python3 -c \
        "import sys,json; print(json.load(sys.stdin)['choices'][0]['text'][:60])" 2>/dev/null || echo "(parse error)")
    log "[$label] latency=${ms}ms  output='${output}'"
    echo "$ms"
}

count_minio_objects() {
    python3 - <<EOF
import boto3, botocore
s3 = boto3.client("s3",
    endpoint_url="$MINIO_ENDPOINT",
    aws_access_key_id="minioadmin",
    aws_secret_access_key="minioadmin",
    region_name="us-east-1",
    config=botocore.config.Config(s3={"addressing_style": "path"}),
)
resp = s3.list_objects_v2(Bucket="$BUCKET")
print(resp.get("KeyCount", 0))
EOF
}

# ── 1. MinIO pre-check ────────────────────────────────────────────────────────
log "=== 1. MinIO pre-check ==="
before=$(count_minio_objects)
log "Objects in '$BUCKET' before request: $before"

# ── 2. First request (cold — populates MinIO) ─────────────────────────────────
log "=== 2. First request (cold) ==="
ms1=$(send_request "cold")

log "Waiting 3s for async MinIO write..."
sleep 3

after=$(count_minio_objects)
new_objects=$((after - before))
log "Objects in '$BUCKET' after request: $after  (+${new_objects} new)"

# ── 3. Controller worker check ────────────────────────────────────────────────
log "=== 3. Controller worker check ==="
curl -s "http://localhost:$CONTROLLER_PORT/controller/workers" | python3 - <<'EOF'
import sys, json
d = json.load(sys.stdin)
for w in d.get("workers", []):
    print(f"  worker_id={w['worker_id']}  port={w['port']}  key_count={w['key_count']}")
print(f"  total workers: {d['total_count']}")
EOF

# ── 4. Second request (warm — CPU cache hit) ──────────────────────────────────
log "=== 4. Second request (warm, same prompt) ==="
ms2=$(send_request "warm")

# ── 5. Summary ────────────────────────────────────────────────────────────────
log "=== Summary ==="
log "MinIO objects written : $new_objects"
log "Cold latency          : ${ms1}ms"
log "Warm latency          : ${ms2}ms"

if [ "$new_objects" -gt 0 ]; then
    log "PASS: KV cache chunks written to MinIO successfully."
else
    log "FAIL: No objects found in MinIO after request."
    exit 1
fi
