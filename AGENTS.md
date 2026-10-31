# Agent Notes: VieNeu-TTS on Vast.ai

## Scope

This work adds a custom Vast.ai Serverless deployment for **VieNeu-TTS v3 Turbo**.
Keep deployment changes in `deploy/vast/` wherever possible. Preserve upstream
API and TTS engine behavior; do not replace them or introduce vLLM, Ollama or
ComfyUI. See `deploy/vast/README.md` for deployment instructions and acceptance checks.

## Deployment Layout

- `deploy/vast/Dockerfile`: Linux AMD64 image, CUDA 12.8 runtime, Python 3.12;
  uses upstream's locked PyTorch/torchaudio 2.8.0 CUDA dependencies.
- `deploy/vast/Dockerfile.dockerignore`: restricted build context; keep secrets,
  local environments and model caches out of images.
- `deploy/vast/runtime.py`: existing API adapter, model configuration, TLS setup,
  process supervision and health checks. `start.sh` is the container entrypoint.
- `deploy/vast/worker.py`: Vast proxy routes, audio forwarding, workload estimates
  and benchmark configuration. `config.py` validates shared environment settings.
- `deploy/vast/requirements.in` and `requirements.txt`: pinned worker dependencies
  and generated hashes. `.env.example` documents runtime configuration.
- `deploy/vast/client.py`: direct API/Vast client, concurrent measurements and
  atomic WAV output. `test_deployment.py` provides CPU-only contract tests.

## Runtime Contracts

- Actual upstream API startup is `python -m apps.openai_speech`.
  `docker/Dockerfile.serve` and `src/vieneu/serve.py` are legacy, not v3 Turbo.
- Reuse `apps/openai_speech.py`. The deployment launcher binds its model factory
  and adds `/_vast/benchmark`; upstream source remains unchanged.
- Default model is `pnnbao-ump/VieNeu-TTS-v3-Turbo`, subfolder `update`.
  Weights download at startup into `HF_HOME`; they are not baked into the image.
- Model selection happens through environment variables at startup. Request
  `model` is an upstream compatibility label, not per-request model switching.
- Full worker mode requires PyTorch/CUDA. Backend stays on loopback port 8000;
  worker uses port 3000 and SDK auxiliary HTTP uses port 3001 by default.
- Preserve `POST /v1/audio/speech`, binary PCM/WAV, SSE, upstream errors,
  `Retry-After`, sample-rate and request-ID headers. Stream with backpressure;
  close upstream connections on cancellation or timeout.
- Vast worker ingress uses the SDK's authenticated request envelope, not a bare
  OpenAI request. The example client uses existing PCM-over-SSE events and checks
  terminal completion. Do not assume Vast's hosted OpenAI proxy supports speech.
- Default to four active streams and four waiting requests on RTX 3060 12GB and
  RTX 3060 Ti 8GB. Keep benchmark concurrency at or below active stream count.
  Tune each GPU separately using measurements, not claimed performance numbers.
- Benchmark complete synthesis, not response headers. The pinned SDK does not
  consume streaming benchmark bodies; `/_vast/benchmark` drains the existing
  speech response before returning JSON. Use character-count workload consistently.
- Emit model readiness only after load, warm-up and successful backend health.
  Vast Ready additionally requires SDK benchmarking and health confirmation.
  Clear stale benchmark/readiness state on every start.

## Dependency And Security Rules

- Model and worker use separate Python environments. Preserve frozen upstream
  `uv.lock`; do not install worker requirements into the model environment.
- Worker uses `vastai-sdk==1.10.0`. Verify changes against official documentation
  and actual SDK interfaces; never invent methods or configuration fields.
- Use `uv --no-config pip` for worker/client dependency operations. Otherwise
  upstream project overrides can inject Linux NVIDIA dependencies and break hashes.
- Production requires TLS, signature verification and Vast-injected instance
  identity. Never substitute an account API key for `MASTER_TOKEN`.
- Leave backend `VIENEU_API_KEY` unset in worker mode; backend is private and the
  worker authenticates ingress. Do not expose backend port 8000 publicly.
- `VAST_LOCAL=1` is test-only: disables TLS/signatures and overrides reporting to
  loopback. Never use it on a public deployment.
- Keep credentials runtime-only and use trusted model repositories. Watermarking
  is deliberately disabled because its optional dependency is not installed.
- Do not push images, create paid workers or change live infrastructure without
  explicit authorization. Do not revert unrelated user changes or commit unasked.

## Validation

Run from this repository root. CPU-only checks need no GPU, weights or Vast key:

```bash
uv venv /tmp/vieneu-vast-tests --python 3.12
uv --no-config pip install --python /tmp/vieneu-vast-tests/bin/python \
  --require-hashes -r deploy/vast/requirements.txt
uv --no-config pip install --python /tmp/vieneu-vast-tests/bin/python \
  fastapi==0.120.2 uvicorn==0.38.0 numpy==2.3.4 \
  python-multipart==0.0.20 httpx==0.28.1 pytest
/tmp/vieneu-vast-tests/bin/python deploy/vast/test_deployment.py -v
PYTHONPATH=src:. /tmp/vieneu-vast-tests/bin/python -m pytest tests/test_openai_speech.py -q
sh -n deploy/vast/start.sh
git diff --check
```

The implementation baseline passed 17 deployment checks and 16 upstream API tests,
including real SDK readiness against a local fake engine. Linux dependency
resolution dry-runs also passed. These are not GPU or live Vast acceptance tests.

AMD64 builds require `docker buildx build --platform linux/amd64` and
`-f deploy/vast/Dockerfile`, with repository root as build context. Apple Silicon
cannot validate NVIDIA inference. Docker build, GPU quality/performance and live
Vast TLS/routing were not verified during implementation; Docker daemon was offline.
Report these limits honestly until validated on an NVIDIA Linux host and Vast.
