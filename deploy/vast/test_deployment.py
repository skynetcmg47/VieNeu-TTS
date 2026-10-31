"""CPU-only contract checks against the pinned Vast SDK and a fake model."""
import asyncio
import base64
from dataclasses import replace
import io
import logging
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, Mock, patch

from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer
import numpy as np

from config import Settings
from runtime import build_app, prepare_transport, supervise
from worker import Worker, create_config, workload
from client import save_audio, sse_audio

ROOT = Path(__file__).resolve().parents[2]


class ConfigurationTests(unittest.TestCase):
    def test_limits(self):
        for environment in ({"VIENEU_MAX_STREAMS": "0"}, {"VIENEU_QUEUE": "0"},
                            {"VIENEU_MAX_STREAMS": "2", "BENCHMARK_CONCURRENCY": "3"},
                            {"VIENEU_DTYPE": "fp16"}, {"PORT": "3000"},
                            {"REQUEST_TIMEOUT": "nan"}, {"VIENEU_MODEL_REPO": ""}):
            with self.subTest(environment=environment), patch.dict(os.environ, environment, clear=True):
                with self.assertRaises(ValueError):
                    Settings.from_env()

    def test_workload_and_benchmark(self):
        with patch.dict(os.environ, {}, clear=True):
            settings = Settings.from_env()
        config = create_config(settings)
        self.assertEqual(config.model_healthcheck_url, "/health")
        speech, benchmark = config.handlers
        self.assertTrue(speech.allow_parallel_requests)
        self.assertIsNone(speech.benchmark_config)
        self.assertEqual(benchmark.benchmark_config.concurrency, settings.streams)
        self.assertIn("VIENEU_VAST_MODEL_READY", config.log_action_config.on_load)
        self.assertEqual(workload({"input": "Xin ch\u00e0o"}), 8)
        for text in (None, 4, [], {}):
            self.assertEqual(workload({"input": text}), 1)

    def test_transport_fails_closed(self):
        for environment in ({}, {"UNSECURED": "true"}, {"USE_SSL": "false"}):
            with patch.dict(os.environ, environment, clear=True):
                with self.assertRaises(ValueError):
                    prepare_transport(Settings.from_env())

    def test_local_requires_explicit_opt_in(self):
        with patch.dict(os.environ, {"VAST_LOCAL": "1", "REPORT_ADDR": "https://run.vast.ai"}, clear=True):
            prepare_transport(Settings.from_env())
            self.assertEqual(os.environ["UNSECURED"], "true")
            self.assertEqual(os.environ["USE_SSL"], "false")
            self.assertEqual(os.environ["REPORT_ADDR"], "http://127.0.0.1:9")

    def test_readiness_only_after_health_and_shutdown(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {
            "LOG_DIR": directory, "WORKER_STATE_DIR": directory,
        }, clear=True):
            settings = Settings.from_env()
            stopped = Mock()
            stopped.is_set.side_effect = [False, False, True]
            processes = [Mock(stdout=io.StringIO(""), pid=123), Mock(stdout=io.StringIO(""), pid=124)]
            for process in processes:
                process.poll.return_value = None

            def check_health(_settings):
                self.assertNotIn("VIENEU_VAST_MODEL_READY", (Path(directory) / "model.log").read_text())
                if not hasattr(check_health, "called"):
                    check_health.called = True
                    raise OSError("not listening yet")
                return {"status": "ok"}

            with patch("runtime.prepare_transport"), patch("runtime.threading.Event", return_value=stopped), \
                    patch("runtime.threading.Thread"), patch("runtime.signal.signal"), \
                    patch("runtime.subprocess.Popen", side_effect=processes), patch("runtime.os.killpg") as kill, \
                    patch("runtime.model_health", side_effect=check_health):
                supervise(settings)
            self.assertIn("VIENEU_VAST_MODEL_READY", (Path(directory) / "model.log").read_text())
            self.assertFalse((Path(directory) / "model-ready").exists())
            self.assertEqual(kill.call_count, 2)

    def test_failed_child_never_ready_and_clears_stale_benchmark(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {
            "LOG_DIR": directory, "WORKER_STATE_DIR": directory,
        }, clear=True):
            settings = Settings.from_env()
            (Path(directory) / ".has_benchmark").write_text("999999")
            (Path(directory) / "model-ready").touch()
            processes = [Mock(stdout=io.StringIO(""), pid=123), Mock(stdout=io.StringIO(""), pid=124)]
            processes[0].poll.return_value = None
            processes[1].poll.return_value = 1
            with patch("runtime.prepare_transport"), patch("runtime.threading.Thread"), \
                    patch("runtime.signal.signal"), patch("runtime.subprocess.Popen", side_effect=processes), \
                    patch("runtime.os.killpg") as kill:
                with self.assertRaisesRegex(RuntimeError, "API or PyWorker exited"):
                    supervise(settings)
            logs = (Path(directory) / "model.log").read_text()
            self.assertIn("VIENEU_VAST_FATAL", logs)
            self.assertNotIn("VIENEU_VAST_MODEL_READY", logs)
            self.assertFalse((Path(directory) / ".has_benchmark").exists())
            self.assertFalse((Path(directory) / "model-ready").exists())
            kill.assert_called_once()


class WorkerTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.environment = patch.dict(os.environ, {
            "CONTAINER_ID": "0", "REPORT_ADDR": "http://127.0.0.1:9",
            "MASTER_TOKEN": "test", "PUBLIC_IPADDR": "127.0.0.1",
            "WORKER_PORT": "3000", "VAST_TCP_PORT_3000": "3000", "UNSECURED": "true",
        }, clear=True)
        self.environment.start()
        self.received = []
        self.benchmark_calls = 0

        async def model(request):
            payload = await request.json()
            self.received.append(payload)
            if payload.get("input") == "bad":
                return web.json_response({"error": "invalid input"}, status=400)
            if payload.get("input") == "busy":
                return web.json_response({"error": "busy"}, status=429, headers={"Retry-After": "1"})
            if payload.get("input") == "down":
                return web.json_response({"error": "down"}, status=503)
            if payload.get("stream_format") == "sse":
                chunks = [b'data: {"type":"speech.audio.delta","audio":"AAE="}\n\n',
                          b'data: {"type":"speech.audio.done","usage":{}}\n\n']
                content_type = "text/event-stream"
            else:
                chunks = [b"RIFF" if payload.get("response_format") == "wav" else b"\x00\xff", b"\x80\x00"]
                content_type = "audio/wav" if payload.get("response_format") == "wav" else "audio/pcm"
            response = web.StreamResponse(headers={
                "Content-Type": content_type, "X-Sample-Rate": "48000", "X-Request-Id": "spk-test",
                "Cache-Control": "no-store", "Connection": "keep-alive, X-Internal", "X-Internal": "secret",
            })
            await response.prepare(request)
            for chunk in chunks:
                await response.write(chunk)
                await asyncio.sleep(0.01)
            await response.write_eof()
            return response

        model_app = web.Application()
        model_app.router.add_post("/v1/audio/speech", model)

        async def benchmark(request):
            await request.json()
            await asyncio.sleep(0.05)
            self.benchmark_calls += 1
            return web.json_response({"audio_bytes": 96})

        async def health(_request):
            return web.json_response({"status": "ok"})

        model_app.router.add_post("/_vast/benchmark", benchmark)
        model_app.router.add_get("/health", health)
        self.model = TestServer(model_app)
        await self.model.start_server()
        self.settings = replace(Settings.from_env(), model_port=self.model.port)
        self.worker = Worker(create_config(self.settings))
        worker_app = web.Application()
        worker_app.add_routes(self.worker.routes)
        self.client = TestClient(TestServer(worker_app, handler_cancellation=True))
        await self.client.start_server()

    async def asyncTearDown(self):
        await self.client.close()
        await self.worker.backend.session.close()
        await self.model.close()
        self.environment.stop()

    async def speech(self, payload):
        return await self.client.post("/v1/audio/speech", json={
            "auth_data": {"cost": "1", "endpoint": "test", "reqnum": 1, "request_idx": 1,
                          "signature": "", "url": "http://127.0.0.1"},
            "payload": payload,
        })

    async def test_binary_headers_and_sse(self):
        for payload, expected, content_type in (
            ({"input": "Xin chao", "response_format": "wav"}, b"RIFF\x80\x00", "audio/wav"),
            ({"input": "Xin chao", "response_format": "pcm"}, b"\x00\xff\x80\x00", "audio/pcm"),
            ({"input": "Xin chao", "stream_format": "sse"}, None, "text/event-stream"),
        ):
            response = await self.speech(payload)
            body = await response.read()
            self.assertEqual(response.status, 200)
            self.assertEqual(response.content_type, content_type)
            self.assertEqual(response.headers["X-Sample-Rate"], "48000")
            self.assertEqual(response.headers["X-Request-Id"], "spk-test")
            self.assertNotIn("X-Internal", response.headers)
            if expected:
                self.assertEqual(body, expected)
            else:
                self.assertIn(b"speech.audio.done", body)
            self.assertEqual(self.received[-1], payload)

    async def test_error_status_and_retry_after(self):
        for text, status in (("bad", 400), ("busy", 429), ("down", 503)):
            response = await self.speech({"input": text})
            self.assertEqual(response.status, status)
            self.assertIn("error", await response.json())
            if status == 429:
                self.assertEqual(response.headers["Retry-After"], "1")

    async def test_auth_not_bypassed(self):
        self.worker.backend.unsecured = False
        self.worker.backend._pubkey = None
        response = await self.speech({"input": "test"})
        self.assertEqual(response.status, 401)
        self.assertEqual(self.received, [])

    async def test_disconnect_closes_upstream(self):
        started = asyncio.Event()

        async def chunks():
            yield b"\x00\x01"
            started.set()
            await asyncio.sleep(10)

        upstream = Mock(status=200, headers={"Content-Type": "audio/pcm"})
        upstream.content.iter_chunked.return_value = chunks()
        downstream = Mock(prepared=True)
        downstream.prepare = AsyncMock()
        downstream.write = AsyncMock()
        request = Mock()
        handler = create_config(self.settings).handlers[0].response_generator
        with patch("worker.web.StreamResponse", return_value=downstream):
            task = asyncio.create_task(handler(request, upstream))
            await started.wait()
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
        upstream.close.assert_called_once()
        request.transport.close.assert_called_once()

    async def test_response_timeout_closes_upstream(self):
        async def chunks():
            await asyncio.sleep(1)
            yield b"\x00\x01"

        upstream = Mock(status=200, headers={"Content-Type": "audio/pcm"})
        upstream.content.iter_chunked.return_value = chunks()
        downstream = Mock(prepared=True)
        downstream.prepare = AsyncMock()
        request = Mock()
        handler = create_config(replace(self.settings, request_timeout=0.01)).handlers[0].response_generator
        with patch("worker.web.StreamResponse", return_value=downstream):
            with self.assertRaises(TimeoutError):
                await handler(request, upstream)
        upstream.close.assert_called_once()
        request.transport.close.assert_called_once()

    async def test_sdk_readiness_lifecycle(self):
        from Crypto.PublicKey import RSA
        public_key = RSA.generate(2048).public_key().export_key()
        reported = asyncio.Event()
        loaded = asyncio.Event()

        async def pubkey(_request):
            return web.Response(body=public_key)

        async def report(request):
            data = await request.json()
            reported.set()
            if data.get("max_perf", 0) > 0:
                loaded.set()
            return web.json_response({})

        control_app = web.Application()
        control_app.router.add_get("/pubkey/", pubkey)
        control_app.router.add_post("/worker_status/", report)
        control_app.router.add_post("/delete_requests/", report)
        control = TestServer(control_app)
        await control.start_server()
        try:
            with tempfile.TemporaryDirectory() as directory:
                model_log = Path(directory) / "model.log"
                model_log.touch()
                with socket.socket() as worker_socket, socket.socket() as http_socket:
                    worker_socket.bind(("127.0.0.1", 0))
                    http_socket.bind(("127.0.0.1", 0))
                    worker_port = worker_socket.getsockname()[1]
                    http_port = http_socket.getsockname()[1]
                environment = {**os.environ, "LOG_DIR": directory,
                               "PORT": str(self.model.port), "WORKER_PORT": str(worker_port),
                               "WORKER_HTTP_PORT": str(http_port),
                               f"VAST_TCP_PORT_{worker_port}": str(worker_port),
                               "REPORT_ADDR": str(control.make_url("/")).rstrip("/"),
                               "BENCHMARK_RUNS": "1", "BENCHMARK_CONCURRENCY": "2",
                               "UNSECURED": "false", "USE_SSL": "false"}
                with (Path(directory) / "sdk.log").open("w") as output:
                    process = subprocess.Popen([sys.executable, str(Path(__file__).with_name("worker.py"))],
                                               cwd=directory, env=environment,
                                               stdout=output, stderr=subprocess.STDOUT)
                    try:
                        await asyncio.wait_for(reported.wait(), timeout=10)
                        self.assertFalse(loaded.is_set())
                        self.assertEqual(self.benchmark_calls, 0)
                        model_log.write_text("VIENEU_VAST_MODEL_READY\n")
                        await asyncio.wait_for(loaded.wait(), timeout=25)
                        self.assertEqual(self.benchmark_calls, 3)
                        self.assertGreater(float((Path(directory) / ".has_benchmark").read_text()), 0)
                    finally:
                        process.terminate()
                        try:
                            await asyncio.to_thread(process.wait, timeout=5)
                        except subprocess.TimeoutExpired:
                            process.kill()
                            await asyncio.to_thread(process.wait)
        finally:
            await control.close()


class APITests(unittest.TestCase):
    def setUp(self):
        from apps import openai_speech as api
        self.api = api
        self.environment = patch.dict(os.environ, {"VIENEU_MAX_STREAMS": "2", "VIENEU_VOICE": "alias"}, clear=True)
        self.environment.start()
        self.settings = Settings.from_env()
        self.settings.apply_model_defaults()
        self.factory = Mock(side_effect=self.fake_model)
        self.factory_patch = patch.object(api, "Vieneu", self.factory)
        self.factory_patch.start()
        self.engine_patch = patch.object(api, "ENGINE", None)
        self.engine_patch.start()
        self.cuda_patch = patch.dict("sys.modules", {"torch": SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: True))})
        self.cuda_patch.start()
        self.app = build_app(self.settings)
        from fastapi.testclient import TestClient as APITestClient
        self.client = APITestClient(self.app)
        self.client.__enter__()

    def tearDown(self):
        self.client.__exit__(None, None, None)
        self.app.router.routes[:] = [route for route in self.app.router.routes if route.path != "/_vast/benchmark"]
        self.cuda_patch.stop()
        self.engine_patch.stop()
        self.factory_patch.stop()
        self.environment.stop()

    def fake_model(self, **kwargs):
        def infer_stream(text, **options):
            time.sleep(0.02)
            yield np.zeros(48, dtype=np.float32)
        return SimpleNamespace(
            backend="pytorch", _preset_voices={"test": {}}, _voice_aliases={"alias": "test"},
            _get_stream_scheduler=lambda: SimpleNamespace(B=kwargs["max_streams"], error=None),
            resolve_voice_name=lambda name: "test" if name in ("alias", "test") else None,
            infer_stream=infer_stream,
        )

    def test_factory_model_selection_voice_and_health(self):
        kwargs = self.factory.call_args.kwargs
        self.assertEqual(kwargs["backbone_repo"], self.settings.model_repo)
        self.assertEqual(kwargs["model_subfolder"], "update")
        self.assertEqual(kwargs["dtype"], "auto")
        self.assertEqual(self.api.ENGINE.tts._default_voice, "test")
        self.assertEqual(self.client.get("/health").json()["max_streams"], 2)

    def test_benchmark_waits_for_audio(self):
        started = time.perf_counter()
        response = self.client.post("/_vast/benchmark", json={"input": "Xin chao"})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["audio_bytes"], 96)
        self.assertGreaterEqual(time.perf_counter() - started, 0.02)
        self.assertEqual(self.api.ENGINE.active, 0)

    def test_upstream_validation_unchanged(self):
        for payload in ({"input": ""}, {"input": "x", "voice": "missing"},
                        {"input": "x", "response_format": "mp3"}):
            response = self.client.post("/v1/audio/speech", json=payload)
            self.assertEqual(response.status_code, 400)

    def test_existing_gate_stays_bounded(self):
        engine = self.api.ENGINE
        for _ in range(self.settings.streams):
            engine.acquire()
        try:
            engine.waiting = engine.max_queue
            response = self.client.post("/v1/audio/speech", json={"input": "Xin chao"})
            self.assertEqual(response.status_code, 429)
            self.assertEqual(response.headers["Retry-After"], "1")
        finally:
            engine.waiting = 0
            for _ in range(self.settings.streams):
                engine.release()
        self.assertEqual(engine.active, 0)


class ClientTests(unittest.IsolatedAsyncioTestCase):
    async def test_sse_and_atomic_output(self):
        async def events(complete):
            yield {"type": "speech.audio.delta", "audio": base64.b64encode(b"\x00\x01").decode()}
            if complete:
                yield {"type": "speech.audio.done", "usage": {"sample_rate": 48000, "output_samples": 1}}
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "speech.wav"
            output.write_bytes(b"previous-good-audio")
            with self.assertRaises(ValueError):
                await save_audio(sse_audio(events(False), 48000), output, 48000)
            self.assertEqual(output.read_bytes(), b"previous-good-audio")
            result = await save_audio(sse_audio(events(True), 48000), output, 48000)
            self.assertEqual(result["bytes"], 2)
            self.assertEqual(output.read_bytes()[:4], b"RIFF")
            self.assertEqual(len(list(Path(directory).iterdir())), 1)


if __name__ == "__main__":
    sys.path[:0] = [str(ROOT), str(ROOT / "src")]
    logging.disable(logging.CRITICAL)
    unittest.main()
