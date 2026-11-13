# Cloud Run: ONNX CPU speech API

GitHub Actions now builds only the `api` target in `docker/Dockerfile.cpu` for
Linux AMD64. No GPU image is built. The image uses ONNX Runtime, without PyTorch,
CUDA, Vast SDK, or model weights. The existing Gradio image remains the default
Dockerfile target for local Web UI builds.

## Publish

Push relevant changes to `main`, or run **Actions > Publish ONNX Image > Run
workflow**. The workflow uses `GITHUB_TOKEN` with `packages: write`, checks CPU
dependencies and API host/port handling without downloading models, then pushes:

```text
ghcr.io/<owner>/<repository>-onnx:sha-<7-character-commit-sha>
ghcr.io/<owner>/<repository>-onnx:latest
```

Names are lowercase. `latest` is updated only from the default branch. No Google
credentials are required: the workflow publishes to GHCR, not to Google Cloud,
and does not deploy a service. Make the GHCR package public before using its URL
directly in Cloud Run. For a private package, use an authenticated Artifact
Registry remote repository or copy the image into Artifact Registry.

## Cloud Run service

- Select the published ONNX image; prefer a digest or commit tag.
- Use CPU only, no GPU. Start with 4 vCPU, 8 GiB RAM, concurrency 1, and maximum
  instances 1; adjust from measured memory use and synthesis latency.
- Keep default container command/arguments. It starts
  `python -m apps.openai_speech`, binds `0.0.0.0`, and reads Cloud Run's `PORT`.
  Set the container port to 8080.
- Keep IAM authentication required. If deliberately allowing public access,
  supply `VIENEU_API_KEY` securely; this API also exposes voice enrollment.
- Configure an HTTP startup probe on `/health`, for example every 10 seconds,
  timeout 10 seconds, failure threshold 24 (240 seconds). The first successful
  health response includes model loading and warm-up. Download/warm-up must fit
  this window. Docker's port exposure alone is not a model readiness check.
- Set request timeout to 300 seconds initially. Streaming audio still needs to
  finish within the request deadline.

Image defaults: `VIENEU_BACKEND=onnx`, `VIENEU_DEVICE=cpu`,
`VIENEU_PRECISION=int8`, `VIENEU_MAX_STREAMS=1`, `VIENEU_QUEUE=1`, and
`VIENEU_WATERMARK=0`. The optional watermark dependency is not installed.

Weights download into `HF_HOME` during API startup. Cloud Run's writable
filesystem is ephemeral and consumes instance RAM, so include downloaded model
files in the memory budget. A warm instance may reuse its cache; a fresh instance
must download again. Enrolled voices are also process-local, not persistent.
The speech route remains `POST /v1/audio/speech`; no Vast envelope is required.

## Local build

```bash
docker buildx build --platform linux/amd64 --target api --load \
  -f docker/Dockerfile.cpu -t vieneu-tts:onnx .
docker run --rm -p 127.0.0.1:8080:8080 \
  -v hf_cache:/home/app/.cache/huggingface vieneu-tts:onnx
```

CI's checks do not synthesize audio or validate live Cloud Run IAM, downloads,
memory, cold starts, or performance. Validate these before production traffic.
