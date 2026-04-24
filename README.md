# pesto

A self-contained experiment workspace for studying KV-cache persistence with
[LMCache](https://github.com/LMCache/LMCache) and [vLLM](https://github.com/vllm-project/vllm),
using [MinIO](https://min.io) as the S3-compatible object-storage backend.

The setup runs two peer vLLM instances that share a common KV-cache store via
LMCache, enabling fault-tolerance experiments: when one instance restarts it
can recover its KV cache from MinIO instead of recomputing from scratch.

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
| [uv](https://docs.astral.sh/uv/) | `curl -LsSf https://astral.sh/uv/install.sh \| sh` |
| CUDA-capable GPU | tested with CUDA 12.x |
| `curl`, `git` | standard system tools |

MinIO is downloaded automatically by `setup.sh` if not found on `$PATH`.

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

This does the following in order:

1. Checks that `uv` is installed and that the `LMCache` submodule is present (auto-initialises if needed).
2. Runs `uv sync` to create `.venv/` and install all uv-managed packages (vllm, boto3, …).
3. Installs LMCache from the bundled source (`./LMCache`) in editable mode with `--no-build-isolation` (required because LMCache's build system depends on the already-installed PyTorch).
4. Compiles LMCache's CUDA/C++ extensions in-place (`build_ext --inplace`).
5. Downloads the MinIO binary to `./bin/minio` if `minio` is not already on `$PATH` (architecture auto-detected: `amd64` or `arm64`).

To point at a different LMCache source tree:

```bash
LMCACHE_SRC=/path/to/other/LMCache ./setup.sh
```

### 3. Start MinIO and create the bucket (first time only)

```bash
./start_minio.sh &
./setup_bucket.sh     # waits for MinIO to be ready, then creates the bucket
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

## Running the smoke test

```bash
./test_minio_cache.sh
```

The script checks four things in order:

1. **vLLM instance 0** responds on `/health`.
2. **LMCache controller** responds and has at least one registered worker.
3. **MinIO bucket** contains KV cache objects. If the bucket is empty the script sends a seeded request to force a write and then re-checks.
4. **End-to-end inference** — sends a completion request and verifies a valid response.

Expected output:

```
[HH:MM:SS] === 1. vLLM instance 0 ===
  [PASS] vLLM-0 is healthy (port 8000)
[HH:MM:SS] === 2. LMCache controller ===
  [PASS] Controller has 1 registered worker(s)
[HH:MM:SS] === 3. MinIO bucket ===
  [PASS] MinIO bucket 'lmcache-bucket' contains 34 object(s)
[HH:MM:SS] === 4. End-to-end inference ===
  [PASS] Inference succeeded in 73ms, output='The'
[HH:MM:SS] === Summary: 4 passed, 0 failed ===
```

## Stopping services

```bash
pkill -f "minio server"
pkill -f "lmcache.v1.api_server"
pkill -f "vllm.entrypoints.openai.api_server"
```
