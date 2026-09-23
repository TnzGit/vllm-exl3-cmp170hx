#!/usr/bin/env python3
"""Measure one idle-server, non-KVMEM C1 prefill configuration cell.

Use the same --prompt-token-ids JSON file for exact prompt identity across
server configurations. Without it, --context selects a deterministic text
prompt; the authoritative token count is still the API usage value.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


FILLER = (
    "The quick brown fox jumps over the lazy dog while the silent panda reads "
    "a book about quantum chromodynamics and semiconductor lithography. "
)
METRICS = {
    "running": "vllm:num_requests_running",
    "waiting": "vllm:num_requests_waiting",
    "preemptions": "vllm:num_preemptions_total",
}


class CellError(RuntimeError):
    """A condition that invalidates the benchmark cell."""


def _prompt(
    context: int, token_ids_path: Path | None
) -> tuple[str | list[int], str]:
    if token_ids_path is not None:
        raw = json.loads(token_ids_path.read_text())
        if isinstance(raw, dict):
            raw = raw.get("prompt_token_ids")
        if not isinstance(raw, list) or not raw:
            raise CellError(
                "prompt-token-ids JSON must be a non-empty list "
                "(or contain prompt_token_ids)"
            )
        if any(
            isinstance(token, bool) or not isinstance(token, int) or token < 0
            for token in raw
        ):
            raise CellError("prompt-token-ids must contain only non-negative integers")
        if len(raw) != context:
            raise CellError(
                f"token-ID prompt length {len(raw)} does not match --context {context}"
            )
        return raw, "token_ids"

    repetitions = max(1, (context * 6 // len(FILLER)) + 2)
    return (FILLER * repetitions)[: context * 6], "deterministic_text"


def _scrape(port: int) -> dict[str, float]:
    try:
        body = urllib.request.urlopen(
            f"http://127.0.0.1:{port}/metrics", timeout=10
        ).read().decode("utf-8", "replace")
    except Exception as exc:
        raise CellError(f"cannot read server metrics: {exc}") from exc

    values: dict[str, float] = {}
    for line in body.splitlines():
        if not line or line.startswith("#"):
            continue
        name, sep, value = line.rpartition(" ")
        if not sep:
            continue
        for key, metric in METRICS.items():
            if name in (metric, metric + "_total") or name.startswith(
                (metric + "{", metric + "_total{")
            ):
                try:
                    values[key] = values.get(key, 0.0) + float(value)
                except ValueError:
                    pass
    missing = set(METRICS) - values.keys()
    if missing:
        raise CellError("required server metrics missing: " + ", ".join(sorted(missing)))
    return values


def _require_idle(metrics: dict[str, float], where: str) -> None:
    if metrics["running"] != 0 or metrics["waiting"] != 0:
        raise CellError(
            f"server busy {where}: running={metrics['running']:g}, "
            f"waiting={metrics['waiting']:g}"
        )


def _request(
    port: int, model: str, prompt: str | list[int], max_tokens: int
) -> dict[str, Any]:
    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "seed": 0,
        "ignore_eos": True,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    started = time.perf_counter()
    ttft_s: float | None = None
    usage: dict[str, Any] | None = None
    output_parts: list[str] = []
    try:
        with urllib.request.urlopen(request, timeout=7200) as response:
            for raw in response:
                raw = raw.strip()
                if not raw:
                    continue
                if raw.startswith(b"data: "):
                    raw = raw[6:]
                if raw == b"[DONE]":
                    break
                try:
                    event = json.loads(raw)
                except (UnicodeDecodeError, json.JSONDecodeError):
                    continue
                if event.get("usage") is not None:
                    usage = event["usage"]
                if event.get("error"):
                    raise CellError(f"server returned error: {event['error']}")
                choice = (event.get("choices") or [{}])[0]
                logprobs = choice.get("logprobs") or {}
                if isinstance(choice.get("text"), str):
                    output_parts.append(choice["text"])
                if ttft_s is None and (choice.get("text") or logprobs.get("tokens")):
                    ttft_s = time.perf_counter() - started
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise CellError(f"completions request failed: {exc}") from exc

    wall_s = time.perf_counter() - started
    if ttft_s is None:
        raise CellError("stream ended without an output token")
    if not isinstance(usage, dict):
        raise CellError("stream did not include usage; include_usage is required")
    try:
        prompt_tokens = int(usage["prompt_tokens"])
        completion_tokens = int(usage["completion_tokens"])
    except (KeyError, TypeError, ValueError) as exc:
        raise CellError("usage is missing prompt_tokens or completion_tokens") from exc
    if prompt_tokens <= 0 or completion_tokens <= 0:
        raise CellError("usage prompt_tokens/completion_tokens must be positive")

    output_text = "".join(output_parts)
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "ttft_s": ttft_s,
        "wall_s": wall_s,
        "output_text_sha256": hashlib.sha256(output_text.encode("utf-8")).hexdigest(),
        "output_text": output_text,
        "prefill_tok_s": prompt_tokens / ttft_s,
        "tpot_s": (wall_s - ttft_s) / (completion_tokens - 1)
        if completion_tokens > 1 else None,
    }


def _one_checked_request(
    *,
    port: int,
    model: str,
    prompt: str | list[int],
    max_tokens: int,
    expected_prompt_tokens: int | None,
    preemptions_before: float,
) -> tuple[dict[str, Any], float]:
    before = _scrape(port)
    _require_idle(before, "before request")
    if before["preemptions"] != preemptions_before:
        raise CellError("preemptions changed between requests")

    sample = _request(port, model, prompt, max_tokens)
    after = _scrape(port)
    _require_idle(after, "after request")
    preemption_delta = after["preemptions"] - before["preemptions"]
    sample["preemption_delta"] = preemption_delta
    if preemption_delta != 0:
        raise CellError(f"request caused {preemption_delta:g} preemption(s)")
    if expected_prompt_tokens is not None and sample["prompt_tokens"] != expected_prompt_tokens:
        raise CellError(
            f"usage.prompt_tokens={sample['prompt_tokens']} does not match "
            f"exact prompt length {expected_prompt_tokens}"
        )
    if sample["completion_tokens"] != max_tokens:
        raise CellError(
            f"usage.completion_tokens={sample['completion_tokens']} does not match "
            f"--max-tokens {max_tokens}"
        )
    return sample, after["preemptions"]


def _median(samples: list[dict[str, Any]], key: str) -> float | None:
    values = [float(sample[key]) for sample in samples if sample.get(key) is not None]
    return statistics.median(values) if values else None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--context", type=int, required=True, help="prompt length in tokens"
    )
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--port", type=int, default=8002)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--prompt-token-ids",
        type=Path,
        help="JSON integer array (or object with prompt_token_ids) for exact prompt identity",
    )
    args = parser.parse_args()
    if args.context < 1 or args.repeats < 1 or args.max_tokens < 2:
        parser.error("--context and --repeats must be positive; --max-tokens must be at least 2")

    result: dict[str, Any] = {
        "schema": 1,
        "benchmark": "non_kvmem_c1_mtp_k3_prefill_scan",
        "context_requested": args.context,
        "repeats_requested": args.repeats,
        "max_tokens_requested": args.max_tokens,
        "port": args.port,
        "prompt_source": None,
        "warmup": None,
        "samples": [],
        "median": None,
        "output_parity": None,
        "status": "INVALID",
        "error": None,
    }
    try:
        prompt, prompt_source = _prompt(args.context, args.prompt_token_ids)
        expected_prompt_tokens = len(prompt) if isinstance(prompt, list) else None
        base = f"http://127.0.0.1:{args.port}"
        model_info = json.load(urllib.request.urlopen(f"{base}/v1/models", timeout=30))
        model = model_info["data"][0]["id"]
        initial = _scrape(args.port)
        _require_idle(initial, "before warmup")
        preemptions = initial["preemptions"]
        result["prompt_source"] = prompt_source
        result["prompt_sha256"] = hashlib.sha256(
            json.dumps(prompt, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        result["model"] = model
        result["warmup"] = {"requested": True, "max_tokens": min(16, args.max_tokens)}

        warmup, preemptions = _one_checked_request(
            port=args.port,
            model=model,
            prompt=prompt,
            max_tokens=min(16, args.max_tokens),
            expected_prompt_tokens=expected_prompt_tokens,
            preemptions_before=preemptions,
        )
        result["warmup"].update(warmup)

        samples: list[dict[str, Any]] = result["samples"]
        for index in range(args.repeats):
            try:
                sample, preemptions = _one_checked_request(
                    port=args.port,
                    model=model,
                    prompt=prompt,
                    max_tokens=args.max_tokens,
                    expected_prompt_tokens=expected_prompt_tokens,
                    preemptions_before=preemptions,
                )
                sample["repeat"] = index + 1
                samples.append(sample)
            except CellError as exc:
                samples.append({"repeat": index + 1, "error": str(exc)})
                raise

        output_hashes = {sample["output_text_sha256"] for sample in samples}
        result["output_parity"] = len(output_hashes) == 1

        result["median"] = {
            "prompt_tokens": _median(samples, "prompt_tokens"),
            "completion_tokens": _median(samples, "completion_tokens"),
            "ttft_s": _median(samples, "ttft_s"),
            "wall_s": _median(samples, "wall_s"),
            "prefill_tok_s": _median(samples, "prefill_tok_s"),
            "tpot_s": _median(samples, "tpot_s"),
        }
        result["status"] = (
            "VALID" if result["output_parity"]
            else "VALID_PERFORMANCE_WITH_OUTPUT_VARIATION"
        )
    except (CellError, OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        result["error"] = str(exc)
    finally:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result, indent=2))
    return 0 if result["status"].startswith("VALID") else 2


if __name__ == "__main__":
    raise SystemExit(main())
