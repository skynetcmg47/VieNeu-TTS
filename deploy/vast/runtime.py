"""Launch the existing speech API and supervise its Vast PyWorker."""
import asyncio
from contextlib import suppress
import json
import os
from pathlib import Path
import signal
import socket
import ssl
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request

from config import Settings

ROOT = Path(__file__).resolve().parents[2]
READY = "VIENEU_VAST_MODEL_READY"
FATAL = "VIENEU_VAST_FATAL"


def build_app(settings):
    from apps import openai_speech as api

    original_factory = api.Vieneu

    def configured_factory(**kwargs):
        if kwargs.get("backend") != "onnx":
            import torch
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA unavailable; refusing silent CPU fallback")
        model = original_factory(
            backbone_repo=settings.model_repo,
            model_subfolder=settings.model_subfolder,
            moss_tokenizer=settings.codec_repo,
            dtype=settings.dtype,
            **kwargs,
        )
        if settings.voice:
            voice = model.resolve_voice_name(settings.voice)
            if voice is None:
                raise ValueError(f"Unknown VIENEU_VOICE: {settings.voice!r}")
            model._default_voice = voice
        return model

    api.Vieneu = configured_factory

    @api.app.post("/_vast/benchmark")
    async def benchmark(req: api.SpeechRequest):
        request = req.model_copy(update={"response_format": "pcm", "stream_format": "audio"})
        response = await asyncio.to_thread(api.speech, request)
        size = 0
        try:
            async with asyncio.timeout(settings.request_timeout):
                async for chunk in response.body_iterator:
                    size += len(chunk)
        finally:
            await response.body_iterator.aclose()
        if not size:
            raise api.HTTPException(503, "Benchmark produced no audio")
        return {"audio_bytes": size, "sample_rate": request.sample_rate}

    return api.app


def model_health(settings):
    with urllib.request.urlopen(f"http://127.0.0.1:{settings.model_port}/health", timeout=3) as response:
        body = json.load(response)
    if body.get("status") != "ok" or body.get("backend") != "pytorch":
        raise RuntimeError("Model health did not confirm healthy PyTorch backend")
    if body.get("max_streams") != settings.streams:
        raise RuntimeError("Model stream limit differs from worker configuration")
    return body


def prepare_transport(settings):
    local = os.environ.get("VAST_LOCAL", "0")
    if local not in {"0", "1"}:
        raise ValueError("VAST_LOCAL must be 0 or 1")
    if local == "1":
        os.environ.update(USE_SSL="false", UNSECURED="true", REPORT_ADDR="http://127.0.0.1:9")
        os.environ.setdefault("CONTAINER_ID", "0")
        os.environ.setdefault("MASTER_TOKEN", "local-only")
        os.environ.setdefault("PUBLIC_IPADDR", "127.0.0.1")
        os.environ.setdefault(f"VAST_TCP_PORT_{settings.worker_port}", str(settings.worker_port))
        return
    if os.environ.get("UNSECURED", "false").lower() != "false":
        raise ValueError("Production requires UNSECURED=false")
    if os.environ.get("USE_SSL", "true").lower() != "true":
        raise ValueError("Production requires USE_SSL=true")
    os.environ.update(USE_SSL="true", UNSECURED="false")
    for name in ("CONTAINER_ID", "MASTER_TOKEN", "PUBLIC_IPADDR", f"VAST_TCP_PORT_{settings.worker_port}"):
        if not os.environ.get(name):
            raise ValueError(f"Vast must inject {name}; do not substitute a placeholder")
    instance_id = int(os.environ["CONTAINER_ID"])
    if instance_id <= 0:
        raise ValueError("CONTAINER_ID must be positive")
    os.umask(0o077)
    subprocess.run([
        "openssl", "req", "-newkey", "rsa:2048", "-nodes", "-sha256",
        "-subj", "/C=US/ST=CA/CN=pyworker.vast.ai/", "-addext", "subjectAltName=IP:0.0.0.0",
        "-addext", "basicConstraints=CA:FALSE",
        "-addext", "keyUsage=nonRepudiation,digitalSignature,keyEncipherment",
        "-keyout", "/etc/instance.key", "-out", "/etc/instance.csr",
    ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    certificate_request = urllib.request.Request(
        f"https://console.vast.ai/api/v0/sign_cert/?instance_id={instance_id}",
        data=Path("/etc/instance.csr").read_bytes(),
        headers={"Content-Type": "application/octet-stream"}, method="POST",
    )
    with urllib.request.urlopen(certificate_request, timeout=60) as response:
        Path("/etc/instance.crt").write_bytes(response.read())
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.load_cert_chain("/etc/instance.crt", "/etc/instance.key")


def supervise(settings):
    settings.apply_model_defaults()
    if os.environ["VIENEU_BACKEND"] != "pytorch" or not os.environ["VIENEU_DEVICE"].startswith("cuda"):
        raise ValueError("Vast workers require VIENEU_BACKEND=pytorch and VIENEU_DEVICE=cuda")
    if os.environ.get("VIENEU_API_KEY"):
        raise ValueError("Leave VIENEU_API_KEY unset: Vast authenticates worker, backend is loopback-only")
    if os.environ["VIENEU_WATERMARK"] != "0":
        raise ValueError("Image omits watermark extra; use VIENEU_WATERMARK=0")
    os.environ.update(HOST="127.0.0.1", PORT=str(settings.model_port),
                      WORKER_PORT=str(settings.worker_port), LOG_DIR=str(settings.log_dir))
    os.environ.setdefault("REPORT_ADDR", "https://run.vast.ai")
    prepare_transport(settings)
    settings.log_dir.mkdir(parents=True, exist_ok=True)
    state_dir = Path(os.environ.get("WORKER_STATE_DIR", "/tmp/vieneu-worker")).resolve()
    state_dir.mkdir(parents=True, exist_ok=True)
    (state_dir / ".has_benchmark").unlink(missing_ok=True)
    ready_file = state_dir / "model-ready"
    ready_file.unlink(missing_ok=True)
    stopped = threading.Event()
    processes = []
    pumps = []
    log_lock = threading.Lock()
    model_log = settings.log_dir / "model.log"
    for name in ("model.log", "worker.log"):
        path = settings.log_dir / name
        if path.exists():
            path.replace(path.with_suffix(".log.previous"))
        path.touch(mode=0o600)

    def marker(message):
        with log_lock, model_log.open("a", encoding="utf-8") as output:
            output.write(message + "\n")
        print(message, flush=True)

    def pump(process, path):
        with path.open("a", encoding="utf-8") as output:
            for line in process.stdout:
                with log_lock:
                    output.write(line)
                    output.flush()
                print(line, end="", flush=True)

    def stop(_signum, _frame):
        stopped.set()

    for signum in (signal.SIGTERM, signal.SIGINT):
        signal.signal(signum, stop)
    try:
        worker_python = os.environ.get("WORKER_PYTHON", "/opt/worker/bin/python")
        for command, name, cwd in (
            ([worker_python, str(Path(__file__).with_name("worker.py"))], "worker.log", state_dir),
            ([sys.executable, str(Path(__file__)), "api"], "model.log", ROOT),
        ):
            process = subprocess.Popen(command, cwd=cwd, stdout=subprocess.PIPE,
                                       stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace", bufsize=1,
                                       start_new_session=True)
            processes.append(process)
            thread = threading.Thread(target=pump, args=(process, settings.log_dir / name), daemon=True)
            thread.start()
            pumps.append(thread)
        deadline = time.monotonic() + settings.startup_timeout
        ready = False
        failures = 0
        while not stopped.is_set():
            if any(process.poll() is not None for process in processes):
                raise RuntimeError("API or PyWorker exited; inspect model.log and worker.log")
            try:
                model_health(settings)
            except (OSError, ValueError, RuntimeError):
                failures += 1
                if ready and failures >= 3:
                    raise RuntimeError("Model failed three consecutive health checks")
                if not ready and time.monotonic() >= deadline:
                    raise TimeoutError("Model download/load/warm-up exceeded STARTUP_TIMEOUT")
            else:
                failures = 0
                if not ready:
                    marker(READY)
                    ready_file.touch(mode=0o600)
                    ready = True
            stopped.wait(5)
    except Exception:
        marker(FATAL)
        raise
    finally:
        ready_file.unlink(missing_ok=True)
        for process in processes:
            if process.poll() is None:
                with suppress(ProcessLookupError):
                    os.killpg(process.pid, signal.SIGTERM)
        for process in processes:
            try:
                process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                with suppress(ProcessLookupError):
                    os.killpg(process.pid, signal.SIGKILL)
                process.wait()
        for thread in pumps:
            thread.join(timeout=2)


def main():
    settings = Settings.from_env()
    command = sys.argv[1] if len(sys.argv) > 1 else "run"
    if command == "api":
        settings.apply_model_defaults()
        sys.path[:0] = [str(ROOT), str(ROOT / "src")]
        import uvicorn
        uvicorn.run(build_app(settings), host=os.environ.get("HOST", "127.0.0.1"),
                    port=settings.model_port, workers=1, log_level="info")
    elif command == "health":
        model_health(settings)
        with socket.create_connection(("127.0.0.1", settings.worker_port), timeout=3):
            pass
        if not (Path(os.environ.get("WORKER_STATE_DIR", "/tmp/vieneu-worker")) / "model-ready").exists():
            raise RuntimeError("Startup not complete")
    elif command == "run":
        supervise(settings)
    else:
        raise ValueError("Expected run, api or health")


if __name__ == "__main__":
    main()
