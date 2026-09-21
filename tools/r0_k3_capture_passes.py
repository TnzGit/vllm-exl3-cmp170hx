#!/usr/bin/env python3
"""Capture a decode/speculation-window trace for the k=3 production Amdahl.

The window is defined by verification passes, NOT by stream chunks: under
speculative decoding a chunk carries several emitted tokens, so a chunk-based
window cannot express "N verification passes".

Flow: stream a request, let it enter steady decode, then start the profiler,
accumulate until the target number of verification passes has elapsed (measured
from the spec counters), stop the profiler, then drain the request. Counters
s sampled so every per-token number uses the authoritative usage count.

usage:
  python3 r0_k3_capture_passes.py --context 4096 --target-passes 14 \
      --out /tmp/cap.json
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.request

FILLER = (
    "The quick brown fox jumps over the lazy dog while the silent panda reads "
    "a book about quantum chromodynamics and semiconductor lithography. "
)


def scrape(port: int, names: dict) -> dict:
    text = urllib.request.urlopen(
        f"http://127.0.0.1:{port}/metrics", timeout=30
    ).read().decode("utf-8", "replace")
    out = {}
    for key, metric in names.items():
        tot = 0.0
        for line in text.splitlines():
            if line.startswith(metric + "{") or line.startswith(metric + " "):
                try:
                    tot += float(line.rsplit(" ", 1)[1])
                except ValueError:
                    pass
        out[key] = tot
    return out


SPEC = {
    "drafts": "vllm:spec_decode_num_drafts_total",
    "draft_tokens": "vllm:spec_decode_num_draft_tokens_total",
    "accepted": "vllm:spec_decode_num_accepted_tokens_total",
}


def build_prompt(target: int) -> str:
    words = FILLER.split()
    reps = max(1, int(target * 0.75) // len(words))
    return " ".join(words * reps)[: target * 6]


def post(port: int, path: str) -> None:
    urllib.request.urlopen(urllib.request.Request(
        f"http://127.0.0.1:{port}{path}", data=b"", method="POST"), timeout=300
    ).read()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8002)
    ap.add_argument("--context", type=int, default=4096)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--target-passes", type=int, default=14)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    base = f"http://127.0.0.1:{args.port}"
    model = json.load(urllib.request.urlopen(
        f"{base}/v1/models", timeout=30))["data"][0]["id"]
    prompt = build_prompt(args.context)
    payload = {"model": model, "prompt": prompt, "max_tokens": args.max_tokens,
               "temperature": 0.0, "seed": 0, "stream": True,
               "stream_options": {"include_usage": True}}

    t0 = time.time()
    resp = urllib.request.urlopen(urllib.request.Request(
        f"{base}/v1/completions", data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"}), timeout=7200)

    before = scrape(args.port, SPEC)
    started = None
    emitted_chunks = 0
    usage_ct = None
    ptok = None
    finish = None
    stream = iter(resp)

    # Let a few chunks pass so prefill is done, then open the window.
    while emitted_chunks < 3:
        raw = next(stream)
        if raw.startswith(b"data: "):
            raw = raw[len(b"data: "):]
        if raw.strip() == b"[DONE]":
            break
        try:
            ev = json.loads(raw)
        except Exception:
            continue
        if (ev.get("choices") or [{}])[0].get("text"):
            emitted_chunks += 1

    win_before = scrape(args.port, SPEC)
    post(args.port, "/start_profile")
    started = time.time()

    for raw in stream:
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
            emitted_chunks += 1
        if ch.get("finish_reason"):
            finish = ch["finish_reason"]
        if ev.get("usage"):
            u = ev["usage"]
            if u.get("prompt_tokens"):
                ptok = u["prompt_tokens"]
            if u.get("completion_tokens"):
                usage_ct = u["completion_tokens"]
        # Stop the window once the target pass count has accumulated.
        now = scrape(args.port, SPEC)
        if now["drafts"] - win_before["drafts"] >= args.target_passes:
            break

    win_after = scrape(args.port, SPEC)
    post(args.port, "/stop_profile")
    window_s = time.time() - started

    # drain
    try:
        for _ in stream:
            pass
    except Exception:
        pass

    passes = win_after["drafts"] - win_before["drafts"]
    out = {
        "context": args.context,
        "prompt_tokens": ptok,
        "window_s": round(window_s, 3),
        "verification_passes_in_window": int(passes),
        "draft_tokens_in_window": int(win_after["draft_tokens"]
                                      - win_before["draft_tokens"]),
        "accepted_in_window": int(win_after["accepted"] - win_before["accepted"]),
        "usage_completion_tokens_total": usage_ct,
        "emitted_chunks_total": emitted_chunks,
        "finish_reason": finish,
        "note": ("per-token normalisation uses usage.completion_tokens; "
                 "per-pass normalisation uses the spec draft counter"),
    }
    print(json.dumps(out, indent=2))
    if args.out:
        json.dump(out, open(args.out, "w"), indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
