# syntax=docker/dockerfile:1.7
FROM ghcr.io/astral-sh/uv:0.10.8 AS uv
FROM python:3.12-slim-bookworm

ARG TARGETARCH
RUN test "$TARGETARCH" = amd64
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_PROJECT_ENVIRONMENT=/opt/model \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    PATH=/opt/model/bin:/usr/local/nvidia/bin:$PATH \
    LD_LIBRARY_PATH=/usr/local/nvidia/lib:/usr/local/nvidia/lib64 \
    PYTHONPATH=/opt/vieneu/src:/opt/vieneu \
    HF_HOME=/cache/huggingface \
    LOG_DIR=/var/log/vieneu \
    WORKER_STATE_DIR=/tmp/vieneu-worker \
    VIENEU_BACKEND=pytorch \
    VIENEU_DEVICE=cuda \
    VIENEU_WATERMARK=0 \
    WORKER_PORT=3000 \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    OMP_NUM_THREADS=4 \
    MKL_NUM_THREADS=4 \
    TOKENIZERS_PARALLELISM=false

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates openssl libsndfile1 libgomp1 ffmpeg \
    && rm -rf /var/lib/apt/lists/*
COPY --from=uv /uv /usr/local/bin/uv
WORKDIR /opt/vieneu
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --extra cuda --no-install-project --no-build --python /usr/local/bin/python3 && \
    /opt/model/bin/python -c 'import torch, torchaudio; assert torch.version.cuda == "12.8", torch.version.cuda'
COPY deploy/vast/requirements.txt /tmp/worker-requirements.txt
RUN --mount=type=cache,target=/root/.cache/uv \
    uv venv /opt/worker --python /usr/local/bin/python3 && \
    uv --no-config pip install --python /opt/worker/bin/python --require-hashes --only-binary :all: -r /tmp/worker-requirements.txt && \
    /opt/worker/bin/python -c 'from vastai import Worker, WorkerConfig; assert Worker and WorkerConfig'
COPY src/ ./src/
COPY apps/ ./apps/
COPY deploy/vast/ ./deploy/vast/
RUN chmod 755 deploy/vast/start.sh && mkdir -p /cache/huggingface /var/log/vieneu
EXPOSE 3000 3001
HEALTHCHECK --interval=30s --timeout=10s --start-period=1800s --retries=3 \
    CMD ["/opt/model/bin/python", "/opt/vieneu/deploy/vast/runtime.py", "health"]
STOPSIGNAL SIGTERM
ENTRYPOINT ["/opt/vieneu/deploy/vast/start.sh"]
