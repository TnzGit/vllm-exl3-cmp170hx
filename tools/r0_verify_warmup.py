#!/usr/bin/env python3
"""Re-verify a suspected first-request warmup artifact with a fresh engine.

A reused-engine sweep compares its opening and closing sentinel requests. When
they disagree, the cheap explanation is first-request warmup (lazy init, CUDA
graph capture, page-cache cold start) rather than drift. This script starts
from a healthy engine and measures the SAME short request N times in a row so
the warmup curve is visible per request instead of only at the two ends.

If the first request is an outlier and the rest are flat, the sentinel
mismatch is explained by warmup and later cells are trustworthy. If the value
keeps moving, the engine is genuinely drifting and reuse was invalid.

usage: python3 r0_verify_warmup.py [--port 8002] [--repeats 8] [--out w.json]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request

PROMPT = "The capital of France is"


def once(port: int, model: str, max_tokens: int = 32) -> dict:
    payload = {
        "model": model,
        "prompt": PROMPT,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "seed": 0,
        "stream": True,
    }
    t0 = time.time()
    ttft = None
    n = 0
    with urllib.request.urlopen(
        urllib.request.Request(
            f"http://127.0.0.1:{port}/v1/completions",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        ),
        timeout=3600,
    ) as resp:
        for raw in resp:
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
            if (ev.get("choices") or [{}])[0].get("text"):
                if ttft is None:
                    ttft = time.time() - t0
                n += 1
    wall = time.time() - t0
    decode = (wall - ttft) if ttft is not None else None
    return {
        "ttft_s": round(ttft, 4) if ttft else None,
        "completion_tokens": n,
        "decode_s": round(decode, 4) if decode else None,
        "ms_per_output_token": round(decode / n * 1000, 3) if decode and n else None,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8002)
    ap.add_argument("--repeats", type=int, default=8)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    base = f"http://127.0.0.1:{args.port}"
    model = json.load(urllib.request.urlopen(f"{base}/v1/models", timeout=30))["data"][0]["id"]

    runs = [once(args.port, model) for _ in range(args.repeats)]
    vals = [r["ms_per_output_token"] for r in runs if r["ms_per_output_token"]]
    for i, r in enumerate(runs):
        print(f"req {i}: ttft={r['ttft_s']} ms/token={r['ms_per_output_token']}")

    if len(vals) < 3:
        print("ERROR: too few successful requests", file=sys.stderr)
        return 1

    first, rest = vals[0], vals[1:]
    tail = rest[1:] if len(rest) > 1 else rest
    tail_spread = (max(tail) - min(tail)) if tail else 0.0
    verdict = {
        "first_request_ms_per_token": first,
        "rest_mean_ms_per_token": round(sum(rest) / len(rest), 3),
        "tail_spread_ms_per_token": round(tail_spread, 3),
        "first_is_outlier": bool(first > sum(rest) / len(rest) * 1.15),
        "tail_flat": bool(tail_spread < 0.10 * (sum(tail) / len(tail))),
    }
    if verdict["first_is_outlier"] and verdict["tail_flat"]:
        verdict["explanation"] = "first-request warmup; later cells are trustworthy"
    elif verdict["tail_flat"]:
        verdict["explanation"] = "no material drift; sentinel gap was not systematic"
    else:
        verdict["explanation"] = "engine drifting; engine reuse was NOT valid"
    print(json.dumps(verdict, indent=2))

    if args.out:
        with open(args.out, "w") as fh:
            json.dump({"runs": runs, "verdict": verdict}, fh, indent=2)
        print(f"written: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
