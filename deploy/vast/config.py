"""Deployment settings shared by the model launcher and PyWorker."""
from dataclasses import dataclass
import math
import os
from pathlib import Path


def number(name, default, minimum, maximum, cast=int):
    value = cast(os.environ.get(name, default))
    if not minimum <= value <= maximum or not math.isfinite(value):
        raise ValueError(f"{name} must be between {minimum} and {maximum}")
    return value


@dataclass(frozen=True)
class Settings:
    model_repo: str
    model_subfolder: str
    codec_repo: str
    dtype: str
    voice: str
    streams: int
    queue: int
    queue_timeout: float
    benchmark_runs: int
    benchmark_concurrency: int
    startup_timeout: float
    request_timeout: float
    model_port: int
    worker_port: int
    log_dir: Path

    @classmethod
    def from_env(cls):
        streams = number("VIENEU_MAX_STREAMS", 4, 1, 32)
        dtype = os.environ.get("VIENEU_DTYPE", "auto")
        if dtype not in {"auto", "float32", "bfloat16", "float16"}:
            raise ValueError("VIENEU_DTYPE must be auto, float32, bfloat16 or float16")
        repo = os.environ.get("VIENEU_MODEL_REPO", "pnnbao-ump/VieNeu-TTS-v3-Turbo")
        codec = os.environ.get("VIENEU_CODEC_REPO", "OpenMOSS-Team/MOSS-Audio-Tokenizer-Nano")
        if not repo.strip() or not codec.strip():
            raise ValueError("Model and codec repositories must not be empty")
        model_port = number("PORT", 8000, 1024, 65535)
        worker_port = number("WORKER_PORT", 3000, 1024, 65534)
        http_port = number("WORKER_HTTP_PORT", worker_port + 1, 1024, 65535)
        if len({model_port, worker_port, http_port}) != 3:
            raise ValueError("Model, worker and auxiliary HTTP ports must be distinct")
        return cls(
            repo, os.environ.get("VIENEU_MODEL_SUBFOLDER", "update"), codec, dtype,
            os.environ.get("VIENEU_VOICE", ""), streams,
            number("VIENEU_QUEUE", streams, 1, 128),
            number("VIENEU_QUEUE_TIMEOUT", 10, 0.1, 300, float),
            number("BENCHMARK_RUNS", 3, 1, 100),
            number("BENCHMARK_CONCURRENCY", streams, 1, streams),
            number("STARTUP_TIMEOUT", 1800, 10, 7200, float),
            number("REQUEST_TIMEOUT", 300, 1, 3600, float),
            model_port, worker_port,
            Path(os.environ.get("LOG_DIR", "/var/log/vieneu")).resolve(),
        )

    def apply_model_defaults(self):
        os.environ["VIENEU_MAX_STREAMS"] = str(self.streams)
        os.environ["VIENEU_QUEUE"] = str(self.queue)
        os.environ["VIENEU_QUEUE_TIMEOUT"] = str(self.queue_timeout)
        os.environ.setdefault("VIENEU_BACKEND", "pytorch")
        os.environ.setdefault("VIENEU_DEVICE", "cuda")
        os.environ.setdefault("VIENEU_WATERMARK", "0")
