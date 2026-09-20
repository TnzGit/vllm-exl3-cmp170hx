#!/usr/bin/env python3
"""C1 context benchmark against a running vLLM OpenAI server (no-draft).

Uses exact-token prompts built from a deterministic filler so every context
cell reports the same completion length. Records TTFT, prefill tok/s,
ms/output-token and output tok/s.

usage: python3 r0_bench_context.py --context 4096 [--port 8002] [--out cell.json]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request

# A repeated, non-repeating filler that tokenizes to a stable count.
FILLER = (
    "The quick brown fox jumps over the lazy dog while the silent panda reads "
    "a book about quantum chromodynamics and semiconductor lithography. "
)


def build_prompt(target_tokens: int) -> str:
    """Approximate a target token count, then report the exact count from usage."""
    words = FILLER.split()
    reps = max(1, int(target_tokens * 0.75) // len(words))
    return " ".join(words * reps)[: target_tokens * 6]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8002)
    ap.add_argument("--context", type=int, required=True)
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--out", default=None)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    base_url = f"http://127.0.0.1:{args.port}"
    models = json.load(urllib.request.urlopen(f"{base_url}/v1/models", timeout=30))
    model = models["data"][0]["id"]

    payload = {
        "model": model,
        "prompt": build_prompt(args.context),
        "max_tokens": args.max_tokens,
        "temperature": 0.0,
        "seed": 0,
        "stream": True,
    }
    req = urllib.request.Request(
        f"{base_url}/v1/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )

    t0 = time.time()
    ttft = None
    n_completion = 0
    finish = None
    prompt_tokens = None
    try:
        with urllib.request.urlopen(req, timeout=3600) as resp:
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
                ch = (ev.get("choices") or [{}])[0]
                if ch.get("text"):
                    if ttft is None:
                        ttft = time.time() - t0
                    n_completion += 1
                if ch.get("finish_reason"):
                    finish = ch["finish_reason"]
                if ev.get("usage") and ev["usage"].get("prompt_tokens"):
                    prompt_tokens = ev["usage"]["prompt_tokens"]
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    wall = time.time() - t0
    if prompt_tokens is None:
        # Fall back to a dedicated tokenization request only if the stream did
        # not carry usage.
        try:
            r = json.load(
                urllib.request.urlopen(
                    urllib.request.Request(
                        f"{base_url}/v1/completions",
                        data=json.dumps(
                            {
                                "model": model,
                                "prompt": payload["prompt"],
                                "max_tokens": 1,
                                "temperature": 0.0,
                            }
                        ).encode(),
                        headers={"Content-Type": "application/json"},
                    ),
                    timeout=3600,
                )
            )
            prompt_tokens = (r.get("usage") or {}).get("prompt_tokens")
        except Exception as exc:
            print(f"WARN: token count failed: {exc}", file=sys.stderr)

    decode = (wall - ttft) if ttft is not None else None
    ms_per_tok = (decode / n_completion * 1000) if decode and n_completion else None
    prefill_tps = (prompt_tokens / ttft) if prompt_tokens and ttft else None
    out_tps = (n_completion / decode) if decode and n_completion else None

    cell = {
        "tag": args.tag,
        "context_target": args.context,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": n_completion,
        "finish_reason": finish,
        "ttft_s": round(ttft, 4) if ttft else None,
        "decode_s": round(decode, 4) if decode else None,
        "wall_s": round(wall, 4),
        "prefill_tok_s": round(prefill_tps, 2) if prefill_tps else None,
        "ms_per_output_token": round(ms_per_tok, 3) if ms_per_tok else None,
        "output_tok_s": round(out_tps, 3) if out_tps else None,
    }
    print(json.dumps(cell))
    if args.out:
        with open(args.out, "w") as fh:
            json.dump(cell, fh, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
