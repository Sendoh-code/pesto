# pesto

A self-contained experiment workspace for studying KV-cache persistence with
[LMCache](https://github.com/LMCache/LMCache) and [vLLM](https://github.com/vllm-project/vllm),
using [MinIO](https://min.io) as the S3-compatible object-storage backend.

The setup runs two peer vLLM instances that share a common KV-cache store via
LMCache, enabling fault-tolerance experiments: when one instance restarts it
can recover its KV cache from MinIO instead of recomputing from scratch.

## Architecture

```
 ┌──────────────┐   ┌──────────────┐
 │  vLLM inst 0 │   │  vLLM inst 1 │   :8000 / :8001
 │  (LMCache)   │   │  (LMCache)   │
 └──────┬───────┘   └──────┬───────┘
        │  ZMQ              │  ZMQ
        └────────┬──────────┘
                 │ :8300 (pull) / :8400 (reply)
      ┌──────────▼──────────┐
      │  LMCache Controller │   :8100
      └─────────────────────┘

        KV chunks → S3 PUT/GET

      ┌─────────────────────┐
      │       MinIO         │   :9000 (API) / :9001 (Console)
      │  bucket: lmcache-   │
      │         bucket      │
      └─────────────────────┘
```

### Port reference

| Service                    | Port(s)        |
|----------------------------|----------------|
| vLLM instance 0            | 8000           |
| vLLM instance 1            | 8001           |
| LMCache Controller (HTTP)  | 8100           |
| LMCache Controller pull    | 8300           |
| LMCache Controller reply   | 8400           |
| MinIO API                  | 9000           |
| MinIO Console              | 9001           |

## Prerequisites

| Dependency | Notes |
|------------|-------|
| Python 3.10 – 3.13 | |
| [uv](https://docs.astral.sh/uv/) | package manager |
| CUDA-capable GPU | tested with CUDA 12.x |
| [MinIO server binary](https://min.io/download) | must be on `$PATH` |

## Installation

### 1. Clone with submodules

```bash
git clone --recurse-submodules <repo-url> pesto
cd pesto
```

If you already cloned without `--recurse-submodules`:

```bash
git submodule update --init
```

### 2. Build the environment

```bash
./setup.sh
```

This does three things in order:

1. Runs `uv sync` to create `.venv/` and install all uv-managed packages
   (vllm, boto3, and their transitive dependencies).
2. Installs LMCache from the bundled source (`./LMCache`) in editable mode
   using `uv pip install -e ./LMCache --no-build-isolation`.
   The `--no-build-isolation` flag is required because LMCache's build system
   depends on the already-installed version of PyTorch.

To point at a different LMCache source tree:

```bash
LMCACHE_SRC=/path/to/other/LMCache ./setup.sh
```

### 3. Start MinIO and create the bucket (first time only)

```bash
./start_minio.sh &
# wait a moment for MinIO to start
./setup_bucket.sh
```

## Starting services

Each script activates the project venv automatically; run each in its own
terminal or as a background job.

```bash
./start_minio.sh &          # object storage
./start_controller.sh &     # LMCache controller
./start_instance0.sh &      # vLLM instance 0 (port 8000)
./start_instance1.sh &      # vLLM instance 1 (port 8001)
```

Wait until both vLLM instances are ready:

```bash
curl http://localhost:8000/health   # → 200
curl http://localhost:8001/health   # → 200
```

## Running the cache test

```bash
./test_minio_cache.sh
```

The script:

1. Records the object count in the MinIO bucket before any request.
2. Sends a ~150-token prompt to vLLM instance 0 (cold request).
3. Waits 3 s for the async MinIO write to complete, then counts new objects.
4. Queries the LMCache Controller to confirm the worker registered and reports
   the correct `key_count`.
5. Sends the same prompt again (warm request) and reports latency difference.
6. Exits 0 on success, 1 if no objects were written to MinIO.

Expected output (approximate):

```
[HH:MM:SS] === 1. MinIO pre-check ===
[HH:MM:SS] Objects in 'lmcache-bucket' before request: 0
[HH:MM:SS] === 2. First request (cold) ===
[HH:MM:SS] [cold] latency=520ms  output='The'
[HH:MM:SS] Waiting 3s for async MinIO write...
[HH:MM:SS] Objects in 'lmcache-bucket' after request: 34  (+34 new)
[HH:MM:SS] === 3. Controller worker check ===
  worker_id=0  port=8050  key_count=34
  total workers: 1
[HH:MM:SS] === 4. Second request (warm, same prompt) ===
[HH:MM:SS] [warm] latency=495ms  output='The'
[HH:MM:SS] === Summary ===
[HH:MM:SS] MinIO objects written : 34
[HH:MM:SS] Cold latency          : 520ms
[HH:MM:SS] Warm latency          : 495ms
[HH:MM:SS] PASS: KV cache chunks written to MinIO successfully.
```

## Stopping services

```bash
pkill -f "minio server"
pkill -f "lmcache.v1.api_server"
pkill -f "vllm.entrypoints.openai.api_server"
```
