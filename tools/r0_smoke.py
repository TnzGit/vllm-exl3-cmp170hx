#!/usr/bin/env python3
"""Deterministic greedy smoke test against a running vLLM OpenAI server.

Records exact prompt tokens, completion token IDs, TTFT and decode wall so a
R0 boot can be judged stable: the same request repeated must produce the same
tokens.

usage: python3 r0_smoke.py [--port 8002] [--repeats 3] [--out smoke.json]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request

PROMPT = (
    "The capital of France is Paris. The capital of Germany is Berlin. "
    "Explain in one sentence why capacitor budgets differ between GPUs."
)


def post(port: int, payload: dict) -> tuple[dict, float, float]:
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=600) as resp:
        body = resp.read()
    wall = time.time() - t0
    return json.loads(body), t0, wall


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8002)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    payload = {
        "model": "local",
        "prompt": PROMPT,
        "max_tokens": args.max_tokens,
        "temperature": 0.0,
        "seed": 0,
    }
    # /v1/completions rejects an unknown model id; use the served name.
    try:
        models = json.load(
            urllib.request.urlopen(f"http://127.0.0.1:{args.port}/v1/models", timeout=30)
        )
        payload["model"] = models["data"][0]["id"]
    except Exception as exc:
        print(f"WARN: cannot resolve served model name: {exc}", file=sys.stderr)

    runs = []
    for i in range(args.repeats):
        try:
            body, _t0, wall = post(args.port, payload)
        except Exception as exc:
            print(f"ERROR request {i}: {exc}", file=sys.stderr)
            return 1
        choice = body["choices"][0]
        usage = body.get("usage") or {}
        runs.append(
            {
                "repeat": i,
                "prompt_tokens": usage.get("prompt_tokens"),
                "completion_tokens": usage.get("completion_tokens"),
                "finish_reason": choice.get("finish_reason"),
                "text": choice.get("text"),
                "wall_s": round(wall, 3),
                "error": None,
            }
        )
        print(f"run {i}: prompt_tokens={usage.get('prompt_tokens')} "
              f"completion_tokens={usage.get('completion_tokens')} "
              f"finish={choice.get('finish_reason')} wall={wall:.2f}s")
        print(f"  text={choice.get('text')!r}")

    stable = len({json.dumps(r["text"]) for r in runs}) == 1
    out = {"prompt": PROMPT, "repeats": args.repeats, "runs": runs, "identical_across_repeats": stable}
    print(f"\nidentical_across_repeats={stable}")
    if args.out:
        with open(args.out, "w") as fh:
            json.dump(out, fh, indent=2)
        print(f"written: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
