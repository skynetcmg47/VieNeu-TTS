"""Save speech atomically or measure concurrent requests without buffering audio."""
import argparse
import asyncio
import base64
import json
import os
from pathlib import Path
import statistics
import tempfile
import time
import wave

import aiohttp

from worker import BENCHMARK_TEXT, workload


async def save_audio(chunks, output, sample_rate):
    temporary = None
    writer = None
    size = 0
    first = None
    started = time.perf_counter()
    try:
        if output:
            with tempfile.NamedTemporaryFile(dir=output.parent, prefix=output.name + ".", suffix=".partial",
                                             delete=False) as handle:
                temporary = Path(handle.name)
            writer = wave.open(str(temporary), "wb")
            writer.setparams((1, 2, sample_rate, 0, "NONE", "not compressed"))
        async for chunk in chunks:
            if not chunk:
                continue
            if first is None:
                first = time.perf_counter() - started
            size += len(chunk)
            if writer:
                writer.writeframesraw(chunk)
        elapsed = time.perf_counter() - started
        if not size or size % 2:
            raise ValueError("Audio empty or truncated mid-sample")
        if writer:
            writer.close()
            writer = None
            temporary.replace(output)
        seconds = size / (2 * sample_rate)
        return {"ttfa_ms": first * 1000, "generation_s": elapsed,
                "audio_s": seconds, "rtf": elapsed / seconds, "bytes": size}
    finally:
        try:
            await chunks.aclose()
        finally:
            if writer:
                writer.close()
            if temporary:
                temporary.unlink(missing_ok=True)


async def sse_audio(events, sample_rate):
    done = False
    size = 0
    try:
        async for event in events:
            if event.get("type") == "speech.audio.delta":
                chunk = base64.b64decode(event["audio"], validate=True)
                size += len(chunk)
                yield chunk
            elif event.get("type") == "speech.audio.done":
                usage = event["usage"]
                if usage["sample_rate"] != sample_rate or usage["output_samples"] * 2 != size:
                    raise ValueError("SSE audio length or sample rate mismatch")
                done = True
                break
        if not done:
            raise ValueError("Speech stream ended without speech.audio.done")
    finally:
        await events.aclose()


async def run(args):
    if args.requests > 1 and args.output:
        raise ValueError("Use --output only for a single request")
    payload = {"model": "vieneu-v3-turbo", "input": args.input,
               "response_format": "pcm", "sample_rate": args.sample_rate}
    if args.voice:
        payload["voice"] = args.voice
    semaphore = asyncio.Semaphore(args.concurrency)
    headers = {"Authorization": f"Bearer {os.environ['VIENEU_API_KEY']}"} if os.environ.get("VIENEU_API_KEY") else {}
    async with aiohttp.ClientSession(headers=headers, timeout=aiohttp.ClientTimeout(total=args.timeout)) as http:
        client = None
        endpoint = None
        try:
            if args.endpoint:
                from vastai import Serverless
                client = Serverless(api_key=os.environ.get("VAST_API_KEY"))
                await client.__aenter__()
                endpoint = await client.get_endpoint(name=args.endpoint)

            async def request():
                async with semaphore:
                    started = time.perf_counter()
                    if endpoint:
                        async def chunks():
                            result = await endpoint.request(
                                "/v1/audio/speech", {**payload, "stream_format": "sse"},
                                cost=int(workload(payload)), stream=True, retry=False, timeout=args.timeout,
                            )
                            if not result.get("ok"):
                                raise RuntimeError(f"Vast returned HTTP {result.get('status')}: {result.get('text')}")
                            async for chunk in sse_audio(result["response"], args.sample_rate):
                                yield chunk
                        stats = await save_audio(chunks(), args.output, args.sample_rate)
                    else:
                        async def chunks():
                            async with http.post(args.base_url.rstrip("/") + "/v1/audio/speech", json=payload) as response:
                                if response.status != 200:
                                    raise RuntimeError(f"HTTP {response.status}: {(await response.text())[:500]}")
                                if response.content_type != "audio/pcm":
                                    raise ValueError(f"Unexpected audio content type: {response.content_type}")
                                async for chunk in response.content.iter_any():
                                    yield chunk
                        stats = await save_audio(chunks(), args.output, args.sample_rate)
                    stats["total_s"] = time.perf_counter() - started
                    return stats

            started = time.perf_counter()
            results = await asyncio.gather(*(request() for _ in range(args.requests)), return_exceptions=True)
            elapsed = time.perf_counter() - started
            successful = [result for result in results if isinstance(result, dict)]
            errors = [str(result) for result in results if isinstance(result, BaseException)]
            report = {"requests": args.requests, "concurrency": args.concurrency,
                      "failures": len(errors), "errors": errors, "elapsed_s": elapsed,
                      "characters_per_s": len(successful) * workload(payload) / elapsed,
                      "results": successful}
            if successful:
                report["median_ttfa_ms"] = statistics.median(result["ttfa_ms"] for result in successful)
                report["median_rtf"] = statistics.median(result["rtf"] for result in successful)
            print(json.dumps(report, indent=2))
            if errors:
                raise SystemExit(1)
        finally:
            if client:
                await client.close()


def positive(value):
    result = int(value)
    if result <= 0:
        raise argparse.ArgumentTypeError("Must be positive")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    route = parser.add_mutually_exclusive_group()
    route.add_argument("--endpoint", help="Vast endpoint name; needs VAST_API_KEY")
    route.add_argument("--base-url", default="http://127.0.0.1:8000", help="Direct model API, not PyWorker")
    parser.add_argument("--input", default=BENCHMARK_TEXT)
    parser.add_argument("--voice", default="")
    parser.add_argument("--sample-rate", type=int, choices=(8000, 16000, 24000, 48000), default=48000)
    parser.add_argument("--requests", type=positive, default=1)
    parser.add_argument("--concurrency", type=positive, default=1)
    parser.add_argument("--timeout", type=positive, default=600)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    if not arguments.input.strip() or len(arguments.input) > 20000:
        parser.error("--input must contain 1-20000 characters")
    asyncio.run(run(arguments))
