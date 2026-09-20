#!/usr/bin/env python3
"""Measure one C1 decode cell (throughput + preemption + Xid-free interval).

Profiler must be OFF: torch.profiler overhead invalidates absolute latency.

usage: python3 r0_ab_cell.py --port 8002 --context 4096 --repeats 3
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request

FILLER = (
    "The quick brown fox jumps over the lazy dog while the silent panda reads "
    "a book about quantum chromodynamics and semiconductor lithography. "
)


def metrics(port: int) -> dict:
    text = urllib.request.urlopen(
        f"http://127.0.0.1:{port}/metrics", timeout=30
    ).read().decode("utf-8", "replace")
    out = {}
    for key, name in (
        ("running", "vllm:num_requests_running"),
        ("waiting", "vllm:num_requests_waiting"),
        ("preempt", "vllm:num_preemptions_total"),
    ):
        for line in text.splitlines():
            if line.startswith(name + "{") or line.startswith(name + " "):
                try:
                    out[key] = out.get(key, 0.0) + float(line.rsplit(" ", 1)[1])
                except ValueError:
                    pass
    return out


def build_prompt(target: int) -> str:
    words = FILLER.split()
    reps = max(1, int(target * 0.75) // len(words))
    return " ".join(words * reps)[: target * 6]


def one(port: int, model: str, prompt: str, max_tokens: int) -> dict:
    payload = {"model": model, "prompt": prompt, "max_tokens": max_tokens,
               "temperature": 0.0, "seed": 0, "stream": True}
    t0 = time.time()
    ttft = None
    n = 0
    ptok = None
    with urllib.request.urlopen(urllib.request.Request(
            f"http://127.0.0.1:{port}/v1/completions",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"}), timeout=3600) as r:
        for raw in r:
            if not raw.strip():
                continue
            if raw.startswith(b"data: "):
                raw = raw[len(b"data: "):]
            if raw.strip() == b"[DONE]":
                break
            try:
                ev = json.loads(raw)
            except Exception:
                continue
            ch = (ev.get("choices") or [{}])[0]
            if ch.get("text"):
                if ttft is None:
                    ttft = time.time() - t0
                n += 1
            if ev.get("usage") and ev["usage"].get("prompt_tokens"):
                ptok = ev["usage"]["prompt_tokens"]
    wall = time.time() - t0
    if ptok is None:
        try:
            probe = json.load(urllib.request.urlopen(urllib.request.Request(
                f"http://127.0.0.1:{port}/v1/completions",
                data=json.dumps({"model": model, "prompt": prompt, "max_tokens": 1,
                                 "temperature": 0.0}).encode(),
                headers={"Content-Type": "application/json"}), timeout=3600))
            ptok = (probe.get("usage") or {}).get("prompt_tokens")
        except Exception:
            pass
    dec = (wall - ttft) if ttft else None
    return {"prompt_tokens": ptok, "completion_tokens": n,
            "ttft_s": round(ttft, 4) if ttft else None,
            "ms_per_output_token": round(dec / n * 1000, 3) if dec and n else None,
            "output_tok_s": round(n / dec, 3) if dec and n else None}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8002)
    ap.add_argument("--context", type=int, default=4096)
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--tag", default="")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    base = f"http://127.0.0.1:{args.port}"
    model = json.load(urllib.request.urlopen(f"{base}/v1/models", timeout=30))["data"][0]["id"]
    prompt = build_prompt(args.context)

    m0 = metrics(args.port)
    if m0.get("running", 0) or m0.get("waiting", 0):
        print("ERROR: engine not idle before cell", file=sys.stderr)
        return 2

    cells = [one(args.port, model, prompt, args.max_tokens) for _ in range(args.repeats)]
    m1 = metrics(args.port)

    ms = sorted(c["ms_per_output_token"] for c in cells if c["ms_per_output_token"])
    tps = sorted(c["output_tok_s"] for c in cells if c["output_tok_s"])
    med_ms = ms[len(ms) // 2] if ms else None
    result = {
        "tag": args.tag,
        "context": args.context,
        "prompt_tokens": cells[0]["prompt_tokens"],
        "repeats": args.repeats,
        "ms_per_output_token_median": med_ms,
        "output_tok_s_median": tps[len(tps) // 2] if tps else None,
        "cells": cells,
        "preemptions_delta": m1.get("preempt", 0) - m0.get("preempt", 0),
        "spread_ms": round(max(ms) - min(ms), 3) if ms else None,
    }
    print(json.dumps(result, indent=2))
    if args.out:
        with open(args.out, "w") as fh:
            json.dump(result, fh, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
