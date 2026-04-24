#!/bin/bash
# Smoke-test: verify all services are up and MinIO has KV cache data.
#
# Checks:
#   1. vLLM instance 0 is healthy
#   2. LMCache controller is reachable and has registered workers
#   3. MinIO bucket contains KV cache objects
#   4. A completion request succeeds end-to-end
#
# Prerequisites: all services must be running.
#   ./start_minio.sh &
#   ./start_controller.sh &
#   ./start_instance0.sh &

set -euo pipefail
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$REPO_DIR/.venv/bin/activate"

VLLM_PORT="${VLLM_PORT:-8000}"
CONTROLLER_PORT="${CONTROLLER_PORT:-8100}"
MINIO_ENDPOINT="http://localhost:9000"
BUCKET="lmcache-bucket"
MODEL="Qwen/Qwen2.5-0.5B"
PASS=0 FAIL=0

PROMPT="The history of artificial intelligence begins in antiquity, with myths, stories, and rumors of artificial beings endowed with intelligence or consciousness by master craftsmen. The seeds of modern AI were planted by classical philosophers who attempted to describe the process of human thinking as the mechanical manipulation of symbols. This work culminated in the invention of the programmable digital computer in the 1940s, a machine based on the abstract essence of mathematical reasoning. This device and the ideas behind it inspired a handful of scientists to begin seriously discussing the possibility of building an electronic brain. The field of AI research was founded at a workshop held on the campus of Dartmouth College during the summer of 1956. Those who attended would become the leaders of AI research for decades. Many of them made predictions that within a generation, machines as intelligent as humans would exist. Millions of dollars were poured into making this vision come true. Eventually it became obvious that commercial developers and government researchers had grossly underestimated the difficulty of the project."

log()  { echo "[$(date +%H:%M:%S)] $*" >&2; }
ok()   { echo "  [PASS] $*" >&2; PASS=$((PASS+1)); }
fail() { echo "  [FAIL] $*" >&2; FAIL=$((FAIL+1)); }

# ── 1. vLLM health ────────────────────────────────────────────────────────────
log "=== 1. vLLM instance 0 ==="
if curl -sf "http://localhost:$VLLM_PORT/health" &>/dev/null; then
    ok "vLLM-0 is healthy (port $VLLM_PORT)"
else
    fail "vLLM-0 is not responding on port $VLLM_PORT"
fi

# ── 2. Controller + workers ───────────────────────────────────────────────────
log "=== 2. LMCache controller ==="
worker_json=$(curl -sf -X POST "http://localhost:$CONTROLLER_PORT/query_worker_info" \
    -H "Content-Type: application/json" \
    -d "{\"instance_id\":\"pesto\",\"worker_ids\":null}" 2>/dev/null || echo "")

if [ -z "$worker_json" ]; then
    fail "Controller not responding on port $CONTROLLER_PORT"
else
    echo "$worker_json" | python3 -c "
import sys, json
d = json.load(sys.stdin)
workers = d.get('worker_infos', [])
for w in workers:
    print(f'  worker_id={w[\"worker_id\"]}  ip={w[\"ip\"]}  port={w[\"port\"]}', file=sys.stderr)
print(len(workers))
" 2>/dev/null
    n_workers=$(echo "$worker_json" | python3 -c "
import sys,json; print(len(json.load(sys.stdin).get('worker_infos',[])))
" 2>/dev/null)
    if [ "${n_workers:-0}" -gt 0 ]; then
        ok "Controller has $n_workers registered worker(s)"
    else
        fail "Controller has no registered workers"
    fi
fi

# ── 3. MinIO bucket ───────────────────────────────────────────────────────────
log "=== 3. MinIO bucket ==="

minio_count() {
    python3 -c "
import boto3, botocore
s3 = boto3.client('s3',
    endpoint_url='$MINIO_ENDPOINT',
    aws_access_key_id='minioadmin',
    aws_secret_access_key='minioadmin',
    region_name='us-east-1',
    config=botocore.config.Config(s3={'addressing_style': 'path'}),
)
resp = s3.list_objects_v2(Bucket='$BUCKET')
print(resp.get('KeyCount', 0))
" 2>/dev/null || echo "error"
}

before=$(minio_count)
if [ "$before" = "error" ]; then
    fail "Could not reach MinIO at $MINIO_ENDPOINT"
else
    if [ "$before" -gt 0 ]; then
        ok "MinIO bucket '$BUCKET' already contains $before object(s)"
    else
        # Bucket is empty — send a unique prompt to force a new KV write
        log "Bucket is empty, sending a seeded request to trigger S3 write..."
        seed="[seed:$(date +%s%N)]"
        curl -s -X POST "http://localhost:$VLLM_PORT/v1/completions" \
            -H "Content-Type: application/json" \
            -d "{\"model\":\"$MODEL\",
                 \"prompt\":$(echo "$seed $PROMPT" | python3 -c 'import sys,json; print(json.dumps(sys.stdin.read()))'),
                 \"max_tokens\":1,\"temperature\":0}" &>/dev/null
        log "Waiting 5s for async MinIO write..."
        sleep 5
        after=$(minio_count)
        if [ "$after" -gt 0 ]; then
            ok "MinIO bucket '$BUCKET' now contains $after object(s) after seeded write"
        else
            fail "MinIO bucket '$BUCKET' still empty after seeded request"
        fi
    fi
fi

# ── 4. End-to-end inference ───────────────────────────────────────────────────
log "=== 4. End-to-end inference ==="
start=$(date +%s%3N)
response=$(curl -sf -X POST "http://localhost:$VLLM_PORT/v1/completions" \
    -H "Content-Type: application/json" \
    -d "{\"model\":\"$MODEL\",
         \"prompt\":$(echo "$PROMPT" | python3 -c 'import sys,json; print(json.dumps(sys.stdin.read()))'),
         \"max_tokens\":1,\"temperature\":0}" 2>/dev/null || echo "")
end=$(date +%s%3N)
ms=$((end - start))

if [ -z "$response" ]; then
    fail "Inference request failed (no response)"
else
    output=$(echo "$response" | python3 -c \
        "import sys,json; print(json.load(sys.stdin)['choices'][0]['text'][:60])" 2>/dev/null || echo "(parse error)")
    ok "Inference succeeded in ${ms}ms, output='${output}'"
fi

# ── Summary ───────────────────────────────────────────────────────────────────
log "=== Summary: $PASS passed, $FAIL failed ==="
[ "$FAIL" -eq 0 ]
