#!/usr/bin/env python3
"""Deterministic greedy parity smoke (token IDs must match the R0 baseline)."""
from __future__ import annotations
import argparse, json, time, urllib.request

PROMPT = "The capital of France is"

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8002)
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--max-tokens", type=int, default=8)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    base = f"http://127.0.0.1:{a.port}"
    model = json.load(urllib.request.urlopen(f"{base}/v1/models", timeout=30))["data"][0]["id"]
    payload = {"model": model, "prompt": PROMPT, "max_tokens": a.max_tokens,
               "temperature": 0.0, "seed": 0, "logprobs": 1}
    outs = []
    for _ in range(a.repeats):
        r = json.load(urllib.request.urlopen(urllib.request.Request(
            f"{base}/v1/completions", data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"}), timeout=300))
        c = r["choices"][0]
        outs.append({"text": c["text"], "tokens": c["logprobs"]["tokens"], "usage": r["usage"]})
    same = len({json.dumps(o["tokens"]) for o in outs}) == 1
    print("text:", repr(outs[0]["text"]))
    print("tokens:", outs[0]["tokens"])
    print("usage:", outs[0]["usage"])
    print("identical:", same)
    if a.out:
        json.dump({"runs": outs, "identical": same}, open(a.out, "w"), indent=2)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
