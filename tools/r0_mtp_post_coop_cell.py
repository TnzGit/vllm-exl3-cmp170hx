#!/usr/bin/env python3
"""Post-COOP MTP qualification cell: throughput + acceptance + verify geometry.

One C1 request, spec-decode counters sampled immediately before and after so
numerator and denominator cover the same interval. Profiler must be OFF; this is
a production-performance harness, not an attribution tool.

usage: python3 r0_mtp_post_coop_cell.py --port 8002 --context 4096 --out cell.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request

SPEC = {
    "drafts": "vllm:spec_decode_num_drafts_total",
    "draft_tokens": "vllm:spec_decode_num_draft_tokens_total",
    "accepted": "vllm:spec_decode_num_accepted_tokens_total",
}
PER_POS = "vllm:spec_decode_num_accepted_tokens_per_pos_total"
REQ = {
    "running": "vllm:num_requests_running",
    "waiting": "vllm:num_requests_waiting",
    "preempt": "vllm:num_preemptions_total",
}
FILLER = (
    "The quick brown fox jumps over the lazy dog while the silent panda reads "
    "a book about quantum chromodynamics and semiconductor lithography. "
)


def scrape(port: int) -> dict:
    text = urllib.request.urlopen(
        f"http://127.0.0.1:{port}/metrics", timeout=30
    ).read().decode("utf-8", "replace")
    out: dict = {}
    per_pos: list[float] = []
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        if line.startswith(PER_POS):
            try:
                per_pos.append(float(line.rsplit(" ", 1)[1]))
            except ValueError:
                pass
            continue
        for key, metric in {**SPEC, **REQ}.items():
            if line.startswith(metric + "{") or line.startswith(metric + " "):
                try:
                    out[key] = out.get(key, 0.0) + float(line.rsplit(" ", 1)[1])
                except ValueError:
                    pass
    if per_pos:
        out["per_pos"] = per_pos
    return out


def build_prompt(target: int) -> str:
    words = FILLER.split()
    reps = max(1, int(target * 0.75) // len(words))
    return " ".join(words * reps)[: target * 6]


def run(port: int, model: str, prompt: str, max_tokens: int) -> dict:
    # stream_options.include_usage gives an authoritative completion token count.
    # Counting stream chunks as tokens is WRONG under speculative decoding: one
    # chunk can carry several accepted tokens, which inflated MTP ms/token by ~2x.
    payload = {"model": model, "prompt": prompt, "max_tokens": max_tokens,
               "temperature": 0.0, "seed": 0, "stream": True,
               "stream_options": {"include_usage": True}}
    t0 = time.time()
    ttft = None
    n = 0            # stream chunks (diagnostic only)
    usage_ct = None  # authoritative completion tokens
    ptok = None
    finish = None
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
            if ch.get("finish_reason"):
                finish = ch["finish_reason"]
            if ev.get("usage"):
                u = ev["usage"]
                if u.get("prompt_tokens"):
                    ptok = u["prompt_tokens"]
                if u.get("completion_tokens"):
                    usage_ct = u["completion_tokens"]
    wall = time.time() - t0
    if ptok is None:
        try:
            probe = json.load(urllib.request.urlopen(urllib.request.Request(
                f"http://127.0.0.1:{port}/v1/completions",
                data=json.dumps({"model": model, "prompt": prompt,
                                 "max_tokens": 1, "temperature": 0.0}).encode(),
                headers={"Content-Type": "application/json"}), timeout=3600))
            ptok = (probe.get("usage") or {}).get("prompt_tokens")
        except Exception:
            pass
    dec = (wall - ttft) if ttft else None
    return {"prompt_tokens": ptok, "chunks": n, "usage_completion_tokens": usage_ct,
            "completion_tokens": usage_ct, "finish_reason": finish,
            "ttft_s": round(ttft, 4) if ttft else None,
            "decode_s": round(dec, 4) if dec else None,
            "ms_per_output_token": round(dec / usage_ct * 1000, 3)
            if dec and usage_ct else None,
            "output_tok_s": round(usage_ct / dec, 3) if dec and usage_ct else None}


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

    m0 = scrape(args.port)
    if m0.get("running", 0) or m0.get("waiting", 0):
        print("ERROR: engine not idle before cell", file=sys.stderr)
        return 2

    # One warm-up request, not counted.
    run(args.port, model, prompt, 16)

    cells = []
    for _ in range(args.repeats):
        b = scrape(args.port)
        r = run(args.port, model, prompt, args.max_tokens)
        a = scrape(args.port)
        d_drafts = a.get("drafts", 0) - b.get("drafts", 0)
        d_dtok = a.get("draft_tokens", 0) - b.get("draft_tokens", 0)
        d_acc = a.get("accepted", 0) - b.get("accepted", 0)
        pp_b = b.get("per_pos") or []
        pp_a = a.get("per_pos") or []
        per_pos = None
        if d_drafts and len(pp_a) == len(pp_b) and pp_b:
            per_pos = [round((x - y) / d_drafts, 4) for x, y in zip(pp_a, pp_b)]
        ct = r["completion_tokens"]
        cells.append({**r,
                      "spec_drafts": int(d_drafts) if d_drafts else None,
                      "spec_draft_tokens": int(d_dtok) if d_dtok else None,
                      "spec_accepted": int(d_acc) if d_acc else None,
                      "accepted_per_pass": round(d_acc / d_drafts, 4) if d_drafts else None,
                      "accepted_per_output": round(d_acc / ct, 4) if d_acc and ct else None,
                      "emitted_per_pass": round(ct / d_drafts, 4) if d_drafts else None,
                      "per_position_acceptance": per_pos,
                      "preemptions_delta": a.get("preempt", 0) - b.get("preempt", 0)})

    m1 = scrape(args.port)
    ms = sorted(c["ms_per_output_token"] for c in cells if c["ms_per_output_token"])
    tps = sorted(c["output_tok_s"] for c in cells if c["output_tok_s"])
    result = {
        "tag": args.tag,
        "context": args.context,
        "prompt_tokens": cells[0]["prompt_tokens"],
        "repeats": args.repeats,
        "ms_per_output_token_median": ms[len(ms) // 2] if ms else None,
        "output_tok_s_median": tps[len(tps) // 2] if tps else None,
        "spread_ms": round(max(ms) - min(ms), 3) if ms else None,
        "accepted_per_pass": cells[-1]["accepted_per_pass"],
        "accepted_per_output": cells[-1]["accepted_per_output"],
        "emitted_per_pass": cells[-1]["emitted_per_pass"],
        "per_position_acceptance": cells[-1]["per_position_acceptance"],
        "cells": cells,
        "idle_after": {"running": m1.get("running", 0), "waiting": m1.get("waiting", 0)},
        "preemptions_delta_total": m1.get("preempt", 0) - m0.get("preempt", 0),
        "round_ms_unavailable": (
            "no low-overhead per-verification-round timer exists; not instrumented "
            "to avoid perturbing production numbers"
        ),
    }
    print(json.dumps(result, indent=2))
    if args.out:
        with open(args.out, "w") as fh:
            json.dump(result, fh, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
