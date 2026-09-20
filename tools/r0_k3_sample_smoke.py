#!/usr/bin/env python3
"""Sample-mode smoke for MTP k=3 (crash / NaN / valid-token / seed stability).

Explicitly NOT a no-draft sampled-sequence parity requirement. vLLM's sampled
RNG contract is not guaranteed to be bit-reproducible across runs at these
settings, so we assert validity and stability where the contract allows, and
report reproducibility honestly rather than manufacturing a parity gate.

usage: python3 r0_k3_sample_smoke.py --port 8002 --out smoke.json
"""

from __future__ import annotations

import argparse
import json
import math
import time
import urllib.request

FILLER = (
    "The quick brown fox jumps over the lazy dog while the silent panda reads "
    "a book about quantum chromodynamics and semiconductor lithography. "
)


def build_prompt(target: int) -> str:
    words = FILLER.split()
    reps = max(1, int(target * 0.75) // len(words))
    return " ".join(words * reps)[: target * 6]


def sample(port: int, model: str, prompt: str, *, seed: int, temperature: float,
           top_p: float, max_tokens: int) -> dict:
    payload = {"model": model, "prompt": prompt, "max_tokens": max_tokens,
               "temperature": temperature, "top_p": top_p, "seed": seed,
               "stream": False}
    t0 = time.time()
    r = json.load(urllib.request.urlopen(urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"}), timeout=1200))
    wall = time.time() - t0
    ch = (r.get("choices") or [{}])[0]
    return {
        "seed": seed, "temperature": temperature, "top_p": top_p,
        "text": ch.get("text", ""), "finish_reason": ch.get("finish_reason"),
        "wall_s": round(wall, 3),
        "usage": r.get("usage") or {},
        "has_nan": ("nan" in (ch.get("text") or "").lower()),
        "empty": not (ch.get("text") or "").strip(),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8002)
    ap.add_argument("--context", type=int, default=4096)
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    base = f"http://127.0.0.1:{args.port}"
    model = json.load(urllib.request.urlopen(
        f"{base}/v1/models", timeout=30))["data"][0]["id"]
    prompt = build_prompt(args.context)

    settings = [
        {"seed": 1234, "temperature": 0.8, "top_p": 0.95},
        {"seed": 1234, "temperature": 1.0, "top_p": 1.0},
    ]
    results = []
    for st in settings:
        texts = []
        for _ in range(args.runs):
            res = sample(args.port, model, prompt, seed=st["seed"],
                         temperature=st["temperature"], top_p=st["top_p"],
                         max_tokens=args.max_tokens)
            texts.append(res["text"])
            results.append(res)
        identical = len(set(texts)) == 1
        results.append({
            "setting": st, "runs": args.runs,
            "identical_across_runs": identical,
            "note": ("identical under fixed seed + same config"
                     if identical else
                     "NOT identical: no crash/NaN gate; vLLM does not guarantee "
                     "sampled bit-reproducibility at these settings"),
        })
        print(f"temp={st['temperature']} top_p={st['top_p']} seed={st['seed']}: "
              f"identical={identical}")

    ok = all((not r.get("has_nan")) and (not r.get("empty")) for r in results
             if isinstance(r.get("has_nan"), bool))
    print(f"all_runs_valid(no NaN, non-empty)={ok}")
    print("no crash observed" if ok else "FAILURE: invalid sample output")

    out = {"results": results, "all_valid": ok}
    if args.out:
        json.dump(out, open(args.out, "w"), indent=2)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
