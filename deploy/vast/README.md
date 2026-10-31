# VieNeu-TTS v3 Turbo: Vast.ai Serverless

Custom NVIDIA GPU image, with the existing VieNeu speech API behind Vast's
PyWorker. No vLLM, Ollama, ComfyUI, new TTS engine, runtime Git clone or runtime
package installation. Default weights download at startup. All deployment code
stays in this directory; upstream source and dependency lock stay unchanged.

## 1. Architecture and verified interfaces

Inspected upstream revision: `04044e1b8f822f946bafb2c88b716459f4164152`.
Upstream startup command is **`python -m apps.openai_speech`**, not
`src/vieneu/serve.py` or `docker/Dockerfile.serve` (those are legacy v1/v2).

```text
Application using Vast SDK
    Vast endpoint discovery / signed routing
    PyWorker :3000, TLS + Vast signature verification
    Existing apps.openai_speech API at 127.0.0.1:8000
    VieNeu v3 Turbo, PyTorch / CUDA, existing continuous batching
```

`runtime.py api` imports that same FastAPI app, supplies model constructor settings
through its existing factory binding, and adds one benchmark route. The model,
speech route, validation, audio conversion, presets, scheduler, and admission gate
are upstream implementations. Default voice uses upstream's resolved preset ID
and `_default_voice` attribute; this small integration point needs rechecking when
updating upstream.

Dependencies: slim Python 3.12 base, with CUDA 12.8 runtime libraries supplied by
the repository's locked torch/torchaudio 2.8.0 CUDA wheels, not a second CUDA base
image. The build checks that both packages import and report CUDA 12.8; GPU
inference still requires validation on an NVIDIA host. Model dependencies use
frozen `uv.lock` and the `cuda` extra, not legacy extras. `vastai-sdk==1.10.0` has a separate, hash-locked
environment; `setuptools` supplies the SDK's Python 3.12 `distutils` compatibility.
SDK interfaces were checked against its published wheel, not guessed from examples.

Worker uses public `Worker`, `WorkerConfig`, `HandlerConfig`, `BenchmarkConfig`,
`LogActionConfig` and the documented response-generator hook. PCM/WAV/SSE are
streamed with backpressure, upstream status and selected end-to-end headers.
Disconnects/timeouts close upstream connections. No audio is decoded as JSON or
buffered in full by the proxy.

**Benchmark correctness:** this SDK starts timing backend HTTP calls but does not
consume their streaming bodies. Benchmarking `/v1/audio/speech` directly would
measure headers rather than synthesis. `/_vast/benchmark` consumes the existing
speech response fully, counts bytes, then returns a small JSON result. Only that
handler has `BenchmarkConfig`; workloads use input Unicode character count for
both benchmark and production requests. Do not point external clients at this
benchmark route. It is still signature-protected on the worker.

Official references, checked October 31, 2026:

- <https://docs.vast.ai/guides/serverless/creating-new-pyworkers>
- <https://github.com/vast-ai/vast-sdk> (also inspect the pinned release wheel)
- <https://docs.vast.ai/guides/templates/creating-templates>
- <https://docs.vast.ai/guides/serverless/setting-up-endpoints>
- <https://docs.vast.ai/guides/serverless/serverless-parameters>
- <https://docs.vast.ai/guides/serverless/workergroup-parameters>
- <https://docs.vast.ai/guides/serverless/openai-compatible-api>

## 2. Build and publish to GHCR

### GitHub Actions (recommended)

The workflow `.github/workflows/vast-image.yml` builds the existing Dockerfile on
a native AMD64 Linux runner and publishes to `ghcr.io/<owner>/<repository>-vast`
(all lowercase). No local Docker daemon, GPU or model weights are required.

1. Push the deployment files and workflow to your GitHub repository. Changes on
   `main` to `deploy/vast/`, `src/`, `apps/`, `pyproject.toml`, `uv.lock` or the
   workflow trigger a build. Pull requests do not publish images.
2. For a manual build, open **Actions > Publish Vast Image > Run workflow** and
   select the desired branch. The workflow uses the built-in `GITHUB_TOKEN` with
   `contents: read` and `packages: write`; no GHCR token secret is needed. Ensure
   repository/organization policies allow Actions to publish packages. If the
   package already exists, grant this repository Actions write access to it.
3. Every build publishes `sha-<full-commit-sha>`. Builds from the default branch
   also publish `latest`; manual builds from other branches do not change it.
   Actions are SHA-pinned and Docker layers use GitHub Actions cache.
4. After the first successful push, open the package's settings and make it
   **Public** for credential-free Vast pulls, or configure private registry
   credentials in Vast. Publishing does not automatically make a package public.
5. Record the digest shown in the build summary and use
   `ghcr.io/<owner>/<repository>-vast@sha256:<digest>` in the Vast template.
   Treat `latest` as a convenience tag, not a production pin.

The workflow only publishes an image; it does not create or update Vast workers.
GPU inference, audio quality and live TLS/routing still need Vast acceptance tests.

### Manual build from Apple Silicon (optional)

Start Docker Desktop first. Run commands from the **VieNeu-TTS repository root**.
Dockerfile-specific ignore rules exclude secrets, model caches, Git history,
tests and unrelated files from the build context.

```bash
docker info
docker buildx create --name vieneu-amd64 --driver docker-container --use
# If builder already exists: docker buildx use vieneu-amd64
docker buildx inspect --bootstrap
export TAG=v3turbo-vast-20261031-2
export GHCR_OWNER=your_lowercase_github_owner
export GHCR_REPOSITORY=your_lowercase_repository
# Set GHCR_TOKEN securely; token needs package write permission.
printf '%s' "$GHCR_TOKEN" | docker login ghcr.io -u "$GHCR_OWNER" --password-stdin
export IMAGE="ghcr.io/$GHCR_OWNER/$GHCR_REPOSITORY-vast:$TAG"
docker buildx build --platform linux/amd64 \
  -f deploy/vast/Dockerfile -t "$IMAGE" --push .
docker buildx imagetools inspect "$IMAGE"
```

Verify `linux/amd64` in the manifest. Prefer a public image initially; GHCR packages
may be private by default. Private images need registry credentials configured in
Vast's template/registry settings, not baked into the image. Record the resulting
image digest and use it, or a never-reused release tag, in the template.

An AMD64 emulated build on a Mac can take substantial time and disk. No GPU or
weights are required during image build. Mac Docker cannot validate NVIDIA CUDA
inference; use a native AMD64 Linux builder if emulation becomes too slow.

## 3. Environment configuration

Use `deploy/vast/.env.example` as the settings list. Docker `--env-file` does not
expand shell variables. Do not commit actual secrets.

| Variable | Default | Behavior |
| --- | --- | --- |
| `VIENEU_MODEL_REPO` | `pnnbao-ump/VieNeu-TTS-v3-Turbo` | HF repo or mounted local v3 Turbo model directory |
| `VIENEU_MODEL_SUBFOLDER` | `update` | Model subfolder; empty string means repo root |
| `VIENEU_CODEC_REPO` | `OpenMOSS-Team/MOSS-Audio-Tokenizer-Nano` | Compatible codec repo/path |
| `VIENEU_VOICE` | empty | Upstream default voice; otherwise validate preset/alias at load |
| `VIENEU_DTYPE` | `auto` | `auto`, `float32`, `bfloat16`, `float16`; start with `auto` |
| `VIENEU_BACKEND` / `VIENEU_DEVICE` | `pytorch` / `cuda` | Required in full worker mode; no silent CPU fallback |
| `VIENEU_MAX_STREAMS` | `4` | Existing continuous batching slots, 1-32 |
| `VIENEU_QUEUE` | stream count | Existing waiting queue, 1-128; zero rejects all requests upstream |
| `VIENEU_QUEUE_TIMEOUT` | `10` | Admission timeout seconds; also worker queue-time threshold |
| `BENCHMARK_RUNS` | `3` | Benchmark rounds, plus one warm-up |
| `BENCHMARK_CONCURRENCY` | stream count | Requests per round; must not exceed stream count |
| `STARTUP_TIMEOUT` | `1800` | Model download/load/warm-up deadline, seconds |
| `REQUEST_TIMEOUT` | `300` | Proxy response and internal benchmark consumption deadline |
| `VIENEU_WATERMARK` | `0` | Required; optional watermark package deliberately omitted |
| `CUDA_VISIBLE_DEVICES` | CUDA default | Use `0` for a one-GPU worker |
| `OMP_NUM_THREADS` / `MKL_NUM_THREADS` | `4` / `4` | Tune to reserved CPU capacity |
| `HF_HOME` | `/cache/huggingface` | Writable cache; reused on the same instance, not across machines |
| `HF_TOKEN` | unset | Only if model access requires it; inject at runtime |
| `PORT` | `8000` | Private model port; full supervisor forces loopback host |
| `WORKER_PORT` / `WORKER_HTTP_PORT` | `3000` / `3001` | SDK TLS port / auxiliary HTTP session-end port |
| `LOG_DIR` | `/var/log/vieneu` | Model and worker logs; one previous run retained |
| `WORKER_STATE_DIR` | `/tmp/vieneu-worker` | Benchmark cache and model health marker; reset each start |
| `USE_SSL` / `UNSECURED` | `true` / `false` in production | Fail closed if insecure settings supplied |
| `REPORT_ADDR` | `https://run.vast.ai` | Engine reporting destination; normally leave unchanged |
| `VAST_LOCAL` | `0` | Explicit test-only setting; disables TLS/signatures and uses local reporting |

Vast must supply **`CONTAINER_ID`, `MASTER_TOKEN`, `PUBLIC_IPADDR`, and
`VAST_TCP_PORT_3000`** (or the mapping for your custom worker port). Do not invent
these values or substitute your personal account API key. Account `VAST_API_KEY`
belongs on the calling application, not the GPU worker. Leave `VIENEU_API_KEY`
unset in worker mode: SDK backend calls do not forward that bearer key; backend
is private loopback, worker ingress uses Vast signatures.

`model` in a speech request is upstream's compatibility label, **not a model
switch**. Constructor variables select weights once per worker. Discovery retains
upstream's `vieneu-v3-turbo` label. Custom weights must match v3 Turbo architecture
and codec. Upstream loads trusted model/tokenizer code; only configure repositories
you trust. Model revision pinning is not exposed by upstream's constructor here;
for immutable weights, mount a vetted local snapshot and compatible codec snapshot.

## 4. Create the custom Vast template

1. Open Vast's Templates page and create a custom template. Set its image to the
   pushed AMD64 image digest or immutable tag. Choose **Entrypoint** launch mode;
   retain image entrypoint `/opt/vieneu/deploy/vast/start.sh` and no command args.
   If template UI requires an explicit command override, use that same absolute
   path. Do not select SSH/Jupyter modes or launch a second copy in On-start.
2. Publish container TCP ports **3000** and **3001** using the template's Docker
   options (`-p 3000:3000 -p 3001:3001`). Vast's actual host mappings may differ;
   runtime-injected `VAST_TCP_PORT_3000` tells the SDK its public port. Port 3001
   is SDK auxiliary HTTP session cleanup, not the model API.
3. Add environment values from the table. Keep `USE_SSL=true`, `UNSECURED=false`,
   `VIENEU_MAX_STREAMS=4`, and `BENCHMARK_CONCURRENCY=4`. Never set `VAST_LOCAL=1`
   on a public instance. Do not publish port 8000.
4. Leave On-start empty. Code and dependencies are baked in, so `PYWORKER_REPO`
   is unnecessary. The official guide's cloning variables apply to its generic
   cloning startup script, which this image replaces.
5. Allocate **60 GB disk**, **16 GB host RAM**, and **at least 4 CPU cores** as
   initial provisioning targets, then revise from measured image/cache/RAM use.
   Cache needs writable storage. New machines still download weights even if a
   previous machine cached them. Avoid mounting over `/opt` or `/opt/vieneu`.
6. Save template and record its template ID/hash. First validate with a single
   worker before enabling broad recruitment.

Startup creates a private TLS key/CSR, requests the instance certificate from
Vast's documented signing endpoint, and validates its key/certificate pair before
launching. Production startup currently runs as root because official SDK TLS
paths are `/etc/instance.key` and `/etc/instance.crt`. Do not make those files
world-readable or copy them into the image. Restrict account/template access.

## 5. Endpoint, GPU groups and scaling

1. In Serverless, use **Advanced setup**, create an endpoint named `vieneu-v3-turbo`.
   Initially set **Minimum Workers=1**, **Max Workers=1**, **Minimum Load=1**,
   **Target Utilization=0.7**. This keeps initial validation small and retains warm
   capacity; inspect actual active worker state rather than assuming a cost cap.
2. Attach the custom template to a worker group. Require **one GPU per worker**,
   verified/rentable hosts, adequate CPU/RAM/disk, and sufficient network download
   speed. Use a host NVIDIA driver supporting CUDA 12.8; prefer a current
   570-series-or-newer driver and validate with an actual CUDA inference smoke test.
3. Start with **RTX 3060 12GB**. Select the exact GPU class in group search filters;
   set group `gpu_ram` to the model's measured memory footprint once known (it is
   a bandwidth/capacity estimate, not the GPU selection filter). For initial
   conservative provisioning use `gpu_ram=8`, and set VRAM selection in the
   hardware search filters independently. Confirm recruited GPU in `nvidia-smi`.
4. Validate **RTX 3060 Ti 8GB** in a separate group/validation endpoint, same four
   streams. This keeps benchmark and quality comparisons attributable to one GPU
   class. Test 8GB availability under load, not only idle model size.
5. Watch Loading progress: weight downloads, model loading, warm-up, then
   `VIENEU_VAST_MODEL_READY`, benchmark completion, first SDK healthcheck, then
   **Ready**. Docker health means model healthy plus worker socket listening;
   it is not proof that Vast benchmarking/routing has completed.
6. After acceptance, set **Max Workers=2** initially, keep Minimum Workers=1 and
   Minimum Load=1, Target Utilization=0.7, **Target Queue Time=1s**, **Max Queue
   Time=10s**. Keep cold capacity parameters at UI defaults initially; total
   workers remain bounded by Max Workers. Tune with actual workload and budget.
   These are starting settings, not latency guarantees.
7. Try eight streams on the 3060 only after tests show sufficient latency/VRAM
   headroom. Change both `VIENEU_MAX_STREAMS` and `BENCHMARK_CONCURRENCY`, restart
   workers, and remeasure. More reserved slots can slow even single requests.
   Keep the Ti profile independent; do not infer its limit from the 12GB card.

Engine queue thresholds, worker queue-time checks, and model's bounded admission
queue are distinct. The worker allows parallel calls because the existing model
scheduler batches them. Model queue overflow/timeout returns `429` with
`Retry-After`; do not hide it in an unbounded application retry loop. Input length
is capped by upstream at 20,000 characters, but clients should use conversational
sentence-sized requests for Vietnamese low-latency inference. Long text may hit
`REQUEST_TIMEOUT`; raise the timeout deliberately or split at sentence boundaries.

Use upstream `auto` dtype and default Vietnamese phonemization/sampling first.
Avoid idle keepalive GPU workloads and privileged clock changes; measure warm and
post-idle latency separately. Larger CUDA batch slots are not free performance.

## 6. Call the serverless endpoint

Install the pinned client environment; it needs no torch:

```bash
uv venv /tmp/vieneu-vast-client --python 3.12
uv --no-config pip install --python /tmp/vieneu-vast-client/bin/python \
  --require-hashes -r deploy/vast/requirements.txt
export VAST_API_KEY=YOUR_CLIENT_API_KEY
/tmp/vieneu-vast-client/bin/python deploy/vast/client.py \
  --endpoint vieneu-v3-turbo --output speech.wav
```

The example calls public `Serverless.get_endpoint()` and `endpoint.request()`
with route `/v1/audio/speech`, character-count cost, and `stream=True`. It requests
upstream's **PCM-over-SSE**, validates base64 and the terminal `speech.audio.done`
sample count, and writes a finalized WAV atomically. Incomplete streams fail
without replacing an existing good output file. Printed JSON excludes keys,
signatures and audio. Pass `--voice` using a preset ID/alias verified on the model.

```bash
/tmp/vieneu-vast-client/bin/python deploy/vast/client.py \
  --endpoint vieneu-v3-turbo --requests 8 --concurrency 4 > benchmark.json
```

No assumption that `openai.vast.ai` supports custom speech routes: official proxy
documentation targets vLLM. PyWorker also requires the SDK's
`{"auth_data": ..., "payload": ...}` envelope; it is not a bare OpenAI HTTP server.
The **backend** retains the raw OpenAI-compatible route. A separately authenticated
gateway would be needed for a stable public OpenAI SDK `base_url`; none is added.

Formats upstream actually supports: `pcm` and `wav`, raw chunked audio or SSE,
mono signed 16-bit little-endian, native 48kHz or resampled 24/16/8kHz. No MP3,
Opus, AAC or FLAC output. `speed`/`instructions` are accepted but ignored upstream.
Raw streamed WAV has unknown-length RIFF/data markers; that is upstream behavior,
not corruption. `client.py` saves a finalized WAV instead.

`GET /v1/models`, `GET /v1/voices`, and voice enrollment remain on the private
backend, not on PyWorker's POST-only configured routes. Uploaded voices are
process-local and disappear on worker replacement; this deployment does not offer
cross-worker voice enrollment or persistence.

## 7. Local tests

### CPU-only contract tests on the Mac

No Docker daemon, GPU, model downloads, paid Vast resources or account key needed:

```bash
uv venv /tmp/vieneu-vast-tests --python 3.12
uv --no-config pip install --python /tmp/vieneu-vast-tests/bin/python \
  --require-hashes -r deploy/vast/requirements.txt
uv --no-config pip install --python /tmp/vieneu-vast-tests/bin/python \
  fastapi==0.120.2 uvicorn==0.38.0 numpy==2.3.4 \
  python-multipart==0.0.20 httpx==0.28.1
/tmp/vieneu-vast-tests/bin/python deploy/vast/test_deployment.py -v
```

Tests use the real installed SDK route factory and a fake audio backend/model.
They check WAV/PCM/SSE bytes, upstream error codes, headers, auth rejection,
cancellation cleanup, readiness ordering, configuration, model binding, benchmark
completion, actual SDK startup/health/reporting against a local fake engine, and
atomic output on truncated SSE. They do not prove CUDA, TLS signing
or a live Vast worker's capacity.

Optional CPU inference smoke test outside Docker, using upstream ONNX backend:

```bash
VIENEU_BACKEND=onnx VIENEU_DEVICE=cpu VIENEU_MAX_STREAMS=1 \
  VIENEU_WATERMARK=0 HOST=127.0.0.1 \
  uv run --frozen python deploy/vast/runtime.py api
```

This is API-only, not production GPU worker mode. First run downloads ONNX weights.
Use the client `--base-url http://127.0.0.1:8000 --output speech.wav` in another
terminal. Model/PyTorch runtime need not be installed in the client environment.

### NVIDIA Linux: direct API smoke test

Requires NVIDIA Container Toolkit and a compatible driver. This command exposes
the API only on host loopback and does not start PyWorker:

```bash
docker run --rm --name vieneu-api --gpus all --stop-timeout 60 --no-healthcheck \
  -e HOST=0.0.0.0 -e VIENEU_MAX_STREAMS=4 \
  -p 127.0.0.1:8000:8000 -v vieneu-hf:/cache/huggingface \
  "$IMAGE" api
```

In another terminal, wait for readiness and inspect presets:

```bash
curl --fail http://127.0.0.1:8000/health
curl --fail http://127.0.0.1:8000/v1/models
curl --fail http://127.0.0.1:8000/v1/voices
curl --fail-with-body -D speech.headers \
  -H 'Content-Type: application/json' \
  -d '{"model":"vieneu-v3-turbo","input":"Xin ch\u00e0o Vi\u1ec7t Nam.","response_format":"wav"}' \
  http://127.0.0.1:8000/v1/audio/speech --output streamed.wav
/tmp/vieneu-vast-client/bin/python deploy/vast/client.py \
  --base-url http://127.0.0.1:8000 --output speech.wav
```

Play `speech.wav`; check intelligibility, Vietnamese tone marks, numbers,
punctuation, proper names and mixed Vietnamese/English. Test all supported sample
rates. Invalid voice, empty input, unsupported formats and invalid sampling bounds
must return the upstream 4xx error, not audio.

### NVIDIA Linux: complete local PyWorker smoke test

Stop the API-only container before reusing its cache. This starts both processes:

```bash
docker run --rm --name vieneu-worker --gpus all --stop-timeout 60 \
  --env-file deploy/vast/.env.example -e VAST_LOCAL=1 \
  -p 127.0.0.1:3000:3000 -v vieneu-hf:/cache/huggingface \
  "$IMAGE"
```

**Security:** `VAST_LOCAL=1` disables worker signatures and TLS. Bind published
ports to `127.0.0.1` only. Never use this setting on Vast or an exposed host.
Local mode overrides `REPORT_ADDR` to a closed loopback port, even when the env
file supplies production settings. SDK reporting/public-key errors are expected:
this smoke test checks proxy traffic, not Vast Ready state. The CPU contract suite
separately tests the real SDK readiness lifecycle against a local fake engine.

Wait for benchmark completion in `docker logs vieneu-worker`, then:

```bash
docker exec vieneu-worker /opt/model/bin/python /opt/vieneu/deploy/vast/runtime.py health
curl --fail-with-body -D worker.headers \
  -H 'Content-Type: application/json' \
  -d '{"auth_data":{"cost":"100","endpoint":"local","reqnum":1,"request_idx":1,"signature":"","url":"http://127.0.0.1:3000"},"payload":{"input":"Xin chao.","response_format":"wav"}}' \
  http://127.0.0.1:3000/v1/audio/speech --output worker.wav
docker stop --time 60 vieneu-worker
```

Test production authentication/TLS only on a real Vast worker with injected
instance variables. Do not emulate production by forging signatures, disabling
certificate verification or setting placeholder identity values.

## 8. Benchmarking, failure drills and operations

Run local or Vast client benchmarks at concurrency **1, 2, 4**, then 8 if needed;
use both short sentences and longer Vietnamese paragraphs. Repeat after at least
30 seconds idle. Report per GPU: TTFA, total latency, audio duration, RTF, failures,
character throughput and GPU peak memory. Client TTFA includes routing/network
wait; upstream model logs also report inference TTFA and RTF.

```bash
for concurrency in 1 2 4; do
  /tmp/vieneu-vast-client/bin/python deploy/vast/client.py \
    --base-url http://127.0.0.1:8000 --requests 12 \
    --concurrency "$concurrency" > "benchmark-$concurrency.json"
done
nvidia-smi --query-gpu=timestamp,name,memory.used,memory.total,utilization.gpu,pstate \
  --format=csv -l 1 > gpu.csv
```

Run `nvidia-smi` sampling in a separate terminal **during** the benchmark; stop it
afterwards. Maximum observed `memory.used` includes CUDA/runtime usage; 1-second
sampling can miss short peaks. For strict allocation peaks, profile PyTorch on
the GPU host. No measured 3060/Ti performance guarantee is asserted here.

Required drills before production:

- Saturate beyond `VIENEU_MAX_STREAMS + VIENEU_QUEUE`; confirm bounded waiting,
  429/Retry-After, recovery and no rising memory. Avoid uncontrolled request floods.
- Disconnect during speech; subsequent requests must still acquire stream slots.
- Stop model child; supervisor must terminate worker and exit nonzero. Simulate
  failed model download, invalid voice and missing GPU; no Ready worker may result.
- Restart with cached and empty weights; each start rotates logs, clears stale
  benchmark score, warms up and benchmarks again. No stale readiness marker.
- Verify unsigned worker requests fail in production and valid SDK requests pass
  with TLS verification enabled. Confirm public port 8000 is unreachable.
- Send a request exceeding configured timeout; fail rather than silently saving
  truncated audio. SSE client requires terminal completion.

Logs: Docker stdout and `/var/log/vieneu/model.log`, `worker.log`; previous startup
retained as `*.log.previous`. Fatal model health failures trigger shutdown after
three failed supervisor probes; SDK also monitors backend health. Failed child
processes end the container. SIGTERM stops both process groups, then force-kills
after grace time. Docker health itself does not restart containers; Vast's worker
lifecycle/container restart policy must do that. Monitor Ready counts, loading
time, non-2xx responses, latency, memory, recruitment failures and billing.

SDK limitation: its generic handler counts a returned HTTP response as transport
success, including a preserved upstream 4xx/5xx. Proxy logs preserve status codes;
do not treat SDK success counts alone as application success. Monitor client HTTP
errors too. Mid-stream failures cannot change already-sent HTTP 200; connection
closes and the SSE client rejects missing completion.

## 9. Production checklist

- [ ] Immutable AMD64 image pushed and pull-tested; registry visibility/credentials verified.
- [ ] Repository revision, image digest, model snapshots, CUDA stack and hashed worker dependencies recorded.
- [ ] Secrets runtime-only; trusted model repos; no `.env` or credentials in build context.
- [ ] Entrypoint mode, correct ports, TLS signing and Vast-injected identity verified.
- [ ] `VAST_LOCAL` absent, `UNSECURED=false`, model port private, client account key absent from worker.
- [ ] CUDA inference and Vietnamese audio quality verified on both 3060 and 3060 Ti.
- [ ] Model cache/storage, host RAM, VRAM headroom and driver compatibility verified under load.
- [ ] Benchmark measures complete audio; Ready observed after benchmark and health checks.
- [ ] Load, invalid input, queue overflow, cancellation, timeout, restart and cold-start drills passed.
- [ ] Scaling limits, one warm worker, billing alerts and application error monitoring configured.
- [ ] Previous immutable image/template retained; rollback swaps group/template back and revalidates Ready.

Dependency maintenance: regenerate worker hashes deliberately with
`uv --no-config pip compile deploy/vast/requirements.in --python-version 3.12 --universal
--generate-hashes --output-file deploy/vast/requirements.txt`, rerun contract tests,
rebuild, and repeat live GPU/TLS acceptance. Never upgrade SDK/dependencies in a
running worker. Update upstream and worker SDK separately, inspecting changed
factory, streaming, authentication and benchmarking interfaces each time.
