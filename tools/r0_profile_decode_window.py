#!/usr/bin/env python3
"""Capture a decode-only vLLM torch-profiler window from a streaming request.

The server must be launched with TORCH_PROFILER_DIR set so vLLM registers
/start_profile and /stop_profile. Profiling starts only after the request has
already emitted a few non-empty stream chunks, avoiding prompt-prefill
attribution in the trace.

Example:
  python tools/r0_profile_decode_window.py \
    --base-url http://127.0.0.1:8002 \
    --prompt-file /tmp/prompt.txt \
    --max-tokens 96 --start-after-chunks 2 --profile-chunks 32
"""

from __future__ import annotations

import argparse
import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


FILLER = (
    "The quick brown fox jumps over the lazy dog while the silent panda reads "
    "a book about quantum chromodynamics and semiconductor lithography. "
)


def _build_prompt(target_tokens: int) -> str:
    words = FILLER.split()
    reps = max(1, int(target_tokens * 0.75) // len(words))
    return " ".join(words * reps)[: target_tokens * 6]


def _headers(api_key: str | None, *, json_body: bool = True) -> dict[str, str]:
    out: dict[str, str] = {}
    if json_body:
        out["Content-Type"] = "application/json"
    if api_key:
        out["Authorization"] = f"Bearer {api_key}"
    return out


def _json_request(
    url: str,
    *,
    api_key: str | None,
    payload: dict[str, Any] | None = None,
    method: str | None = None,
) -> dict[str, Any]:
    data = None if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request(
        url,
        data=data,
        headers=_headers(api_key),
        method=method or ("POST" if payload is not None else "GET"),
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        raw = resp.read()
    if not raw:
        return {}
    return json.loads(raw)


def _control(base_url: str, action: str, api_key: str | None) -> None:
    url = f"{base_url.rstrip('/')}/{action}"
    req = urllib.request.Request(
        url,
        data=b"{}",
        headers=_headers(api_key),
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        if resp.status != 200:
            raise RuntimeError(f"{action} returned HTTP {resp.status}")


def _resolve_model(base_url: str, api_key: str | None) -> str:
    obj = _json_request(
        f"{base_url.rstrip('/')}/v1/models",
        api_key=api_key,
    )
    data = obj.get("data") or []
    if not data:
        raise RuntimeError("/v1/models returned no models")
    model = data[0].get("id")
    if not model:
        raise RuntimeError("/v1/models first entry has no id")
    return str(model)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8002")
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("--prompt-file", type=Path)
    group.add_argument("--context", type=int)
    ap.add_argument("--model", default=None)
    ap.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY"))
    ap.add_argument("--max-tokens", type=int, default=96)
    ap.add_argument("--start-after-chunks", type=int, default=2)
    ap.add_argument("--profile-chunks", type=int, default=32)
    ap.add_argument("--timeout", type=float, default=600.0)
    args = ap.parse_args()

    if args.max_tokens <= 0:
        raise SystemExit("--max-tokens must be > 0")
    if args.start_after_chunks < 0 or args.profile_chunks <= 0:
        raise SystemExit("chunk controls must be non-negative / positive")

    if args.prompt_file is not None:
        prompt = args.prompt_file.read_text(encoding="utf-8")
    else:
        assert args.context is not None
        prompt = _build_prompt(args.context)
    model = args.model or _resolve_model(args.base_url, args.api_key)

    payload = {
        "model": model,
        "prompt": prompt,
        "temperature": 0,
        "max_tokens": args.max_tokens,
        "stream": True,
        "stream_options": {"include_usage": True},
        "ignore_eos": True,
    }
    req = urllib.request.Request(
        f"{args.base_url.rstrip('/')}/v1/completions",
        data=json.dumps(payload).encode(),
        headers=_headers(args.api_key),
        method="POST",
    )

    chunks = 0
    output_chars = 0
    profiling = False
    stopped = False
    profile_start_t = None
    profile_stop_t = None
    usage: dict[str, Any] | None = None
    t0 = time.perf_counter()

    try:
        with urllib.request.urlopen(req, timeout=args.timeout) as resp:
            for raw_line in resp:
                line = raw_line.decode("utf-8", errors="replace").strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                if not data:
                    continue
                event = json.loads(data)
                if event.get("usage"):
                    usage = event["usage"]
                choices = event.get("choices") or []
                text = choices[0].get("text", "") if choices else ""
                if not text:
                    continue

                chunks += 1
                output_chars += len(text)

                if (
                    not profiling
                    and chunks > args.start_after_chunks
                ):
                    _control(args.base_url, "start_profile", args.api_key)
                    profiling = True
                    profile_start_t = time.perf_counter()
                    print(
                        f"profile_start chunk={chunks} "
                        f"t={profile_start_t - t0:.6f}s",
                        flush=True,
                    )
                    continue

                if (
                    profiling
                    and not stopped
                    and chunks
                    >= args.start_after_chunks + args.profile_chunks
                ):
                    _control(args.base_url, "stop_profile", args.api_key)
                    stopped = True
                    profile_stop_t = time.perf_counter()
                    print(
                        f"profile_stop chunk={chunks} "
                        f"t={profile_stop_t - t0:.6f}s",
                        flush=True,
                    )
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise SystemExit(f"HTTP {exc.code}: {body}") from exc
    finally:
        if profiling and not stopped:
            try:
                _control(args.base_url, "stop_profile", args.api_key)
                stopped = True
                profile_stop_t = time.perf_counter()
            except Exception as exc:
                print(f"WARNING: stop_profile failed: {exc}", flush=True)

    elapsed = time.perf_counter() - t0
    profile_s = (
        None
        if profile_start_t is None or profile_stop_t is None
        else profile_stop_t - profile_start_t
    )
    print(
        json.dumps(
            {
                "model": model,
                "context_target": args.context,
                "chunks": chunks,
                "output_chars": output_chars,
                "elapsed_s": elapsed,
                "profile_window_s": profile_s,
                "start_after_chunks": args.start_after_chunks,
                "profile_chunks": args.profile_chunks,
                "usage": usage,
            },
            indent=2,
            sort_keys=True,
        )
    )
    if profile_start_t is None:
        raise SystemExit("request ended before profiling could start")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
