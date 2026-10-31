"""Vast proxy configuration; all synthesis stays in apps.openai_speech."""
import asyncio
import logging

from aiohttp import web
from vastai import Worker, WorkerConfig, HandlerConfig, BenchmarkConfig, LogActionConfig

from config import Settings

LOG = logging.getLogger("vieneu.vast")
BENCHMARK_TEXT = (
    "Xin ch\u00e0o, \u0111\u00e2y l\u00e0 h\u1ec7 th\u1ed1ng chuy\u1ec3n v\u0103n b\u1ea3n th\u00e0nh gi\u1ecdng n\u00f3i ti\u1ebfng Vi\u1ec7t. "
    "H\u00f4m nay tr\u1eddi \u0111\u1eb9p, ch\u00fang ta c\u00f9ng b\u1eaft \u0111\u1ea7u m\u1ed9t ng\u00e0y l\u00e0m vi\u1ec7c m\u1edbi."
)
HOP_HEADERS = {"connection", "keep-alive", "proxy-authenticate", "proxy-authorization",
               "te", "trailer", "transfer-encoding", "upgrade", "content-length"}


def workload(payload):
    text = payload.get("input")
    return float(max(1, len(text))) if isinstance(text, str) else 1.0


def create_config(settings):
    def benchmark_payload():
        payload = {"input": BENCHMARK_TEXT, "response_format": "pcm", "sample_rate": 48000}
        if settings.voice:
            payload["voice"] = settings.voice
        return payload

    async def audio_response(client_request, model_response):
        excluded = HOP_HEADERS | {
            name.strip().lower() for name in model_response.headers.get("Connection", "").split(",")
        }
        headers = [(name, value) for name, value in model_response.headers.items()
                   if name.lower() not in excluded]
        response = web.StreamResponse(status=model_response.status, headers=headers)
        try:
            async with asyncio.timeout(settings.request_timeout):
                await response.prepare(client_request)
                async for chunk in model_response.content.iter_chunked(16384):
                    await response.write(chunk)
                await response.write_eof()
            if model_response.status >= 400:
                LOG.warning("Speech backend returned HTTP %d", model_response.status)
            return response
        except (Exception, asyncio.CancelledError):
            if response.prepared and client_request.transport is not None:
                client_request.transport.close()
            raise
        finally:
            model_response.close()

    return WorkerConfig(
        model_server_url="http://127.0.0.1",
        model_server_port=settings.model_port,
        model_log_file=str(settings.log_dir / "model.log"),
        model_healthcheck_url="/health",
        handlers=[
            HandlerConfig(
                route="/v1/audio/speech", allow_parallel_requests=True,
                max_queue_time=settings.queue_timeout, workload_calculator=workload,
                response_generator=audio_response,
            ),
            HandlerConfig(
                route="/_vast/benchmark", allow_parallel_requests=True,
                max_queue_time=settings.queue_timeout, workload_calculator=workload,
                benchmark_config=BenchmarkConfig(
                    generator=benchmark_payload, runs=settings.benchmark_runs,
                    concurrency=settings.benchmark_concurrency, do_warmup=True,
                ),
            ),
        ],
        log_action_config=LogActionConfig(
            on_load=["VIENEU_VAST_MODEL_READY"],
            on_error=["VIENEU_VAST_FATAL"],
        ),
    )


if __name__ == "__main__":
    Worker(create_config(Settings.from_env())).run()
