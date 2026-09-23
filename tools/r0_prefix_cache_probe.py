#!/usr/bin/env python3
"""Read-only serial APC probe for an already-running vLLM OpenAI server.

The probe sends the exact same token-ID prompt repeatedly, samples vLLM's
prefix-cache counters around each request, and fails unless the final repeat
records cached tokens. It never starts a server or modifies runtime files.

Examples:
  python tools/r0_prefix_cache_probe.py --port 8002 \
      --prompt-tokens-file prompt_ids.json --out probe.json
  python tools/r0_prefix_cache_probe.py --port 8002 \
      --deterministic-prompt 'Stable prompt text for the cache probe' --out probe.json

Prompt token files may be a JSON integer array or whitespace/comma-separated
integer IDs. The deterministic-text mode obtains IDs from the server's
read-only /tokenize endpoint, then sends those IDs on every completion request.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


PREFIX_METRICS = {
    "queries": "vllm:prefix_cache_queries",
    "hits": "vllm:prefix_cache_hits",
}
BUSY_METRICS = {
    "running": "vllm:num_requests_running",
    "waiting": "vllm:num_requests_waiting",
}


class ProbeFailure(RuntimeError):
    pass


def parse_metrics(payload: str) -> dict[str, float]:
    """Parse and aggregate relevant Prometheus samples, including labels."""
    wanted = {**PREFIX_METRICS, **BUSY_METRICS}
    values: dict[str, float] = {}
    for raw_line in payload.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        fields = line.split()
        if len(fields) < 2:
            continue
        sample_name = fields[0].split("{", 1)[0]
        for key, metric_name in wanted.items():
            if sample_name not in (metric_name, metric_name + "_total"):
                continue
            try:
                value = float(fields[-1])
            except ValueError:
                continue
            if math.isfinite(value):
                values[key] = values.get(key, 0.0) + value
            break
    return values


def check_required_metrics(metrics: dict[str, float]) -> None:
    missing = sorted(set(PREFIX_METRICS) - metrics.keys())
    if missing:
        names = ", ".join(PREFIX_METRICS[key] for key in missing)
        raise ProbeFailure(f"metrics absent: required prefix-cache metric(s): {names}")
    missing_busy = sorted(set(BUSY_METRICS) - metrics.keys())
    if missing_busy:
        names = ", ".join(BUSY_METRICS[key] for key in missing_busy)
        raise ProbeFailure(f"metrics absent: cannot verify server idleness ({names})")


def assert_idle(metrics: dict[str, float], where: str) -> None:
    check_required_metrics(metrics)
    running = metrics["running"]
    waiting = metrics["waiting"]
    if running > 0 or waiting > 0:
        raise ProbeFailure(
            f"busy server {where}: requests_running={running:g}, "
            f"requests_waiting={waiting:g}"
        )


def evaluate_hit_evidence(deltas: list[dict[str, float]]) -> None:
    """Require an observed lookup and cached-token hit on the final repeat."""
    if len(deltas) < 2:
        raise ProbeFailure("at least two serial prompt requests are required")
    final = deltas[-1]
    queries = final.get("queries", 0.0)
    hits = final.get("hits", 0.0)
    if queries > 0 and hits <= 0:
        raise ProbeFailure(
            "prefix-cache queries increased on the repeated prompt but hits=0"
        )
    if queries <= 0 or hits <= 0:
        raise ProbeFailure(
            "no measured prefix-cache hit on the final repeated prompt "
            f"(queries_delta={queries:g}, hits_delta={hits:g})"
        )


def _request(
    base_url: str,
    path: str,
    payload: dict[str, Any],
    timeout: float,
) -> dict[str, Any]:
    request = urllib.request.Request(
        f"{base_url}{path}",
        data=json.dumps(payload, separators=(",", ":")).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.load(response)
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        raise ProbeFailure(f"{path} request failed: {exc}") from exc


def _fetch_metrics(base_url: str, timeout: float) -> dict[str, float]:
    try:
        with urllib.request.urlopen(f"{base_url}/metrics", timeout=timeout) as response:
            payload = response.read().decode("utf-8", "replace")
    except (urllib.error.URLError, TimeoutError) as exc:
        raise ProbeFailure(f"metrics endpoint unavailable: {exc}") from exc
    metrics = parse_metrics(payload)
    check_required_metrics(metrics)
    return metrics


def _wait_for_idle(
    base_url: str, timeout: float, wait_seconds: float, where: str
) -> dict[str, float]:
    deadline = time.monotonic() + wait_seconds
    while True:
        metrics = _fetch_metrics(base_url, timeout)
        if metrics["running"] <= 0 and metrics["waiting"] <= 0:
            return metrics
        if time.monotonic() >= deadline:
            assert_idle(metrics, where)
        time.sleep(min(0.1, max(0.0, deadline - time.monotonic())))


def _wait_for_prefix_cache_metrics(
    base_url: str,
    timeout: float,
    wait_seconds: float,
    before: dict[str, float],
    *,
    require_hit: bool,
    where: str,
) -> dict[str, float]:
    """Wait for this request's cache counters to reach the metrics endpoint."""
    deadline = time.monotonic() + wait_seconds
    while True:
        metrics = _fetch_metrics(base_url, timeout)
        assert_idle(metrics, where)
        deltas = {
            key: metrics[key] - before[key]
            for key in PREFIX_METRICS
        }
        if any(value < 0 for value in deltas.values()):
            raise ProbeFailure("prefix-cache counters decreased/reset during the probe")

        queries_propagated = deltas["queries"] > 0
        hits_propagated = not require_hit or deltas["hits"] > 0
        if queries_propagated and hits_propagated:
            return metrics

        if time.monotonic() >= deadline:
            waiting_for = ["query counter"] if not queries_propagated else []
            if not hits_propagated:
                waiting_for.append("hit counter")
            raise ProbeFailure(
                f"prefix-cache metrics did not propagate {', '.join(waiting_for)} "
                f"{where} within {wait_seconds:g}s "
                f"(queries_delta={deltas['queries']:g}, "
                f"hits_delta={deltas['hits']:g})"
            )
        time.sleep(min(0.1, max(0.0, deadline - time.monotonic())))


def _read_token_ids(path: Path) -> list[int]:
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ProbeFailure(f"cannot read prompt token file {path}: {exc}") from exc
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        parsed = re.split(r"[\s,]+", raw.strip()) if raw.strip() else []
    if not isinstance(parsed, list) or not parsed:
        raise ProbeFailure("prompt token file must contain a non-empty integer list")
    if any(isinstance(token, bool) or not isinstance(token, int) or token < 0 for token in parsed):
        raise ProbeFailure("prompt token file contains a non-integer or negative token ID")
    return parsed


def _resolve_prompt(args: argparse.Namespace, base_url: str, model: str) -> list[int]:
    if args.prompt_tokens_file is not None:
        tokens = _read_token_ids(args.prompt_tokens_file)
    else:
        result = _request(
            base_url,
            "/tokenize",
            {"model": model, "prompt": args.deterministic_prompt},
            args.timeout,
        )
        tokens = result.get("tokens")
        if not isinstance(tokens, list) or not tokens:
            raise ProbeFailure("/tokenize did not return a non-empty token-ID list")
        if any(isinstance(token, bool) or not isinstance(token, int) or token < 0 for token in tokens):
            raise ProbeFailure("/tokenize returned invalid token IDs")
    if len(tokens) < args.min_prompt_tokens:
        raise ProbeFailure(
            f"prompt has {len(tokens)} tokens; need at least "
            f"{args.min_prompt_tokens} to test a full hybrid cache block"
        )
    return tokens


def _read_sse_completion(
    base_url: str,
    model: str,
    token_ids: list[int],
    args: argparse.Namespace,
) -> dict[str, Any]:
    payload = {
        "model": model,
        "prompt": token_ids,
        "max_tokens": args.max_tokens,
        "temperature": 0,
        "seed": 0,
        "ignore_eos": True,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    request = urllib.request.Request(
        f"{base_url}/v1/completions",
        data=json.dumps(payload, separators=(",", ":")).encode("utf-8"),
        headers={"Content-Type": "application/json", "Accept": "text/event-stream"},
        method="POST",
    )
    start = time.monotonic()
    ttft: float | None = None
    output_parts: list[str] = []
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    try:
        with urllib.request.urlopen(request, timeout=args.timeout) as response:
            for raw in response:
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    event = json.loads(data)
                except json.JSONDecodeError:
                    continue
                if event.get("error"):
                    raise ProbeFailure(f"completion endpoint returned error: {event['error']}")
                usage = event.get("usage") or {}
                if isinstance(usage.get("prompt_tokens"), int):
                    prompt_tokens = usage["prompt_tokens"]
                if isinstance(usage.get("completion_tokens"), int):
                    completion_tokens = usage["completion_tokens"]
                for choice in event.get("choices") or []:
                    text = choice.get("text")
                    if isinstance(text, str):
                        if text and ttft is None:
                            ttft = time.monotonic() - start
                        output_parts.append(text)
    except ProbeFailure:
        raise
    except (urllib.error.URLError, TimeoutError) as exc:
        raise ProbeFailure(f"completion request failed: {exc}") from exc

    output_text = "".join(output_parts)
    if ttft is None:
        raise ProbeFailure("completion produced no text; TTFT is unavailable")
    if completion_tokens is None or prompt_tokens is None:
        raise ProbeFailure("stream omitted prompt/completion token usage")
    if prompt_tokens != len(token_ids) or completion_tokens != args.max_tokens:
        raise ProbeFailure(
            f"unexpected token usage: prompt={prompt_tokens}, "
            f"completion={completion_tokens}"
        )
    return {
        "ttft_seconds": round(ttft, 6),
        "prompt_tokens_usage": prompt_tokens,
        "completion_tokens_usage": completion_tokens,
        "output_text_sha256": hashlib.sha256(output_text.encode("utf-8")).hexdigest(),
        "output_text_characters": len(output_text),
    }


def run_probe(args: argparse.Namespace) -> dict[str, Any]:
    base_url = f"http://127.0.0.1:{args.port}"
    try:
        with urllib.request.urlopen(f"{base_url}/v1/models", timeout=args.timeout) as response:
            models = json.load(response).get("data", [])
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, AttributeError) as exc:
        raise ProbeFailure(f"cannot identify model at /v1/models: {exc}") from exc
    if not models or not isinstance(models[0].get("id"), str):
        raise ProbeFailure("/v1/models returned no usable model ID")
    model = models[0]["id"]
    token_ids = _resolve_prompt(args, base_url, model)

    records: list[dict[str, Any]] = []
    deltas: list[dict[str, float]] = []
    for index in range(args.repeats):
        before = _fetch_metrics(base_url, args.timeout)
        assert_idle(before, f"before request {index + 1}")
        completion = _read_sse_completion(base_url, model, token_ids, args)
        _wait_for_idle(
            base_url, args.timeout, args.idle_timeout, f"after request {index + 1}"
        )
        after = _wait_for_prefix_cache_metrics(
            base_url,
            args.timeout,
            args.metrics_timeout,
            before,
            require_hit=index == args.repeats - 1,
            where=f"after request {index + 1}",
        )
        delta = {
            key: after[key] - before[key]
            for key in PREFIX_METRICS
        }
        if any(value < 0 for value in delta.values()):
            raise ProbeFailure("prefix-cache counters decreased/reset during the probe")
        records.append({
            "request_index": index + 1,
            **completion,
            "prefix_cache_queries_delta": delta["queries"],
            "prefix_cache_hits_delta": delta["hits"],
        })
        deltas.append(delta)

    evaluate_hit_evidence(deltas)
    if len({record["output_text_sha256"] for record in records}) != 1:
        raise ProbeFailure("greedy output changed across identical prompt requests")
    return {
        "status": "PASS",
        "model": model,
        "prompt_token_count": len(token_ids),
        "prompt_token_ids_sha256": hashlib.sha256(
            json.dumps(token_ids, separators=(",", ":")).encode("ascii")
        ).hexdigest(),
        "repeats": args.repeats,
        "requests": records,
        "evidence": "final identical prompt had positive prefix-cache query and hit token deltas",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8002)
    prompt_group = parser.add_mutually_exclusive_group(required=True)
    prompt_group.add_argument("--prompt-tokens-file", type=Path)
    prompt_group.add_argument("--deterministic-prompt", help="text tokenized by the running service")
    parser.add_argument("--repeats", type=int, default=2, help="identical serial requests; minimum 2")
    parser.add_argument("--max-tokens", type=int, default=8)
    parser.add_argument("--min-prompt-tokens", type=int, default=2048)
    parser.add_argument("--timeout", type=float, default=30.0, help="per HTTP operation timeout")
    parser.add_argument("--idle-timeout", type=float, default=5.0)
    parser.add_argument(
        "--metrics-timeout",
        type=float,
        default=30.0,
        help="maximum wait for prefix-cache counters to reach /metrics",
    )
    parser.add_argument("--out", type=Path, help="write the JSON report to this path")
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("--port must be between 1 and 65535")
    if args.repeats < 2:
        parser.error("--repeats must be at least 2")
    if args.max_tokens < 1:
        parser.error("--max-tokens must be positive")
    if args.min_prompt_tokens < 2:
        parser.error("--min-prompt-tokens must be at least 2")
    if args.timeout <= 0 or args.idle_timeout < 0 or args.metrics_timeout <= 0:
        parser.error(
            "--timeout and --metrics-timeout must be positive "
            "(--idle-timeout may be zero)"
        )

    try:
        report = run_probe(args)
        exit_code = 0
    except ProbeFailure as exc:
        report = {"status": "FAIL", "error": str(exc)}
        exit_code = 1

    rendered = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.out:
        try:
            args.out.write_text(rendered, encoding="utf-8")
        except OSError as exc:
            print(f"ERROR: cannot write report to {args.out}: {exc}", file=sys.stderr)
            return 2
    print(rendered, end="")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
