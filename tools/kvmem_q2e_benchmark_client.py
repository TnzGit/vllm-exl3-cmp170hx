#!/usr/bin/env python3
"""Benchmark one exact-token Q2E context through the streaming completions API."""

from __future__ import annotations

import argparse
import json
import time
import urllib.request
from pathlib import Path
from typing import Any, Iterable


def _codes_in_order(text: str, codes: Iterable[str]) -> bool:
    offset = 0
    for code in codes:
        found = text.find(code, offset)
        if found < 0:
            return False
        offset = found + len(code)
    return True


def _metrics(
    *, prompt_tokens: int, completion_tokens: int, ttft_s: float, wall_s: float
) -> dict[str, float | int | None]:
    decode_tokens = max(0, completion_tokens - 1)
    decode_s = max(0.0, wall_s - ttft_s)
    return {
        "prefill_s": ttft_s,
        "prefill_tok_s": prompt_tokens / ttft_s if ttft_s > 0 else None,
        "decode_tokens": decode_tokens,
        "decode_s": decode_s,
        "decode_tok_s": decode_tokens / decode_s
        if decode_tokens and decode_s > 0 else None,
        "decode_ms_per_token": decode_s / decode_tokens * 1000.0
        if decode_tokens and decode_s > 0 else None,
    }


def _stream_request(url: str, payload: dict[str, Any]) -> tuple[list[dict], float, float]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    started = time.perf_counter()
    first_token_s: float | None = None
    events: list[dict] = []
    with urllib.request.urlopen(request, timeout=7200) as response:
        for raw in response:
            raw = raw.strip()
            if not raw:
                continue
            if raw.startswith(b"data: "):
                raw = raw[6:]
            if raw == b"[DONE]":
                break
            event = json.loads(raw)
            events.append(event)
            choice = (event.get("choices") or [{}])[0]
            logprobs = choice.get("logprobs") or {}
            if first_token_s is None and (
                choice.get("text") or logprobs.get("tokens")
            ):
                first_token_s = time.perf_counter() - started
    wall_s = time.perf_counter() - started
    if first_token_s is None:
        raise RuntimeError("stream did not contain an output token")
    return events, first_token_s, wall_s


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8002)
    parser.add_argument("--case", type=Path, required=True)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--allow-semantic-failure", action="store_true")
    args = parser.parse_args()

    case = json.loads(args.case.read_text())
    prompt = [int(token) for token in case["prompt_token_ids"]]
    expected_codes = [
        str(row["code"])
        for row in case.get("target_facts", [])
        if row.get("code") is not None
    ]
    base = f"http://127.0.0.1:{args.port}"
    model = json.load(
        urllib.request.urlopen(f"{base}/v1/models", timeout=30)
    )["data"][0]["id"]
    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": args.max_tokens,
        "temperature": 0.0,
        "seed": 0,
        "ignore_eos": True,
        "stream": True,
        "stream_options": {"include_usage": True},
        "logprobs": 1,
    }
    events, ttft_s, wall_s = _stream_request(
        f"{base}/v1/completions", payload
    )
    usage: dict[str, Any] = {}
    finish_reason = None
    text_parts: list[str] = []
    token_pieces: list[str] = []
    for event in events:
        if event.get("usage"):
            usage = event["usage"]
        choice = (event.get("choices") or [{}])[0]
        text_parts.append(str(choice.get("text") or ""))
        logprobs = choice.get("logprobs") or {}
        token_pieces.extend(str(token) for token in logprobs.get("tokens") or [])
        if choice.get("finish_reason") is not None:
            finish_reason = choice["finish_reason"]

    prompt_tokens = int(usage.get("prompt_tokens", -1))
    completion_tokens = int(usage.get("completion_tokens", -1))
    text = "".join(text_parts)
    semantic_ok = _codes_in_order(text, expected_codes)
    count_ok = prompt_tokens == len(prompt) and completion_tokens == args.max_tokens
    result = {
        "schema": 1,
        "case": case["name"],
        "context_limit": int(case["context_limit"]),
        "prompt_tokens": prompt_tokens,
        "requested_completion_tokens": args.max_tokens,
        "completion_tokens": completion_tokens,
        "finish_reason": finish_reason,
        "expected_codes": expected_codes,
        "target_codes_in_order": semantic_ok,
        "count_exact": count_ok,
        "ignore_eos": True,
        "wall_s": wall_s,
        **_metrics(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            ttft_s=ttft_s,
            wall_s=wall_s,
        ),
        "text": text,
        "token_pieces": token_pieces,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    if not count_ok:
        return 3
    if not semantic_ok and not args.allow_semantic_failure:
        return 4
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
