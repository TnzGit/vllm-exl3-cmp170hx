#!/usr/bin/env python3
"""MTP acceptance + throughput cell against a running vLLM server.

Samples the Prometheus spec-decode counters immediately before and after ONE
request so numerator and denominator cover the same interval. C1 is required
for that guarantee; the tool asserts no other request is in flight.

usage:
  python3 r0_mtp_cell.py --port 8002 --context 4096 [--out cell.json]
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import urllib.request

COUNTERS = {
    "drafts": "vllm:spec_decode_num_drafts_total",
    "draft_tokens": "vllm:spec_decode_num_draft_tokens_total",
    "accepted": "vllm:spec_decode_num_accepted_tokens_total",
}
PER_POS = "vllm:spec_decode_num_accepted_tokens_per_pos_total"
REQUEST_COUNTERS = {
    "requests_running": "vllm:num_requests_running",
    "requests_waiting": "vllm:num_requests_waiting",
    "preemptions": "vllm:num_preemptions_total",
}

FILLER = (
    "The quick brown fox jumps over the lazy dog while the silent panda reads "
    "a book about quantum chromodynamics and semiconductor lithography. "
)


def scrape(port: int) -> dict:
    text = urllib.request.urlopen(
        f"http://127.0.0.1:{port}/metrics", timeout=30
    ).read().decode("utf-8", "replace")
    out: dict[str, float] = {}
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
        for key, metric in {**COUNTERS, **REQUEST_COUNTERS}.items():
            if line.startswith(metric + "{") or line.startswith(metric + " "):
                try:
                    out[key] = out.get(key, 0.0) + float(line.rsplit(" ", 1)[1])
                except ValueError:
                    pass
    if per_pos:
        out["accepted_per_pos"] = per_pos  # type: ignore[assignment]
    return out


def build_prompt(target_tokens: int) -> str:
    words = FILLER.split()
    reps = max(1, int(target_tokens * 0.75) // len(words))
    return " ".join(words * reps)[: target_tokens * 6]


def run_request(port: int, model: str, prompt: str, max_tokens: int) -> dict:
    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "seed": 0,
        "stream": True,
    }
    t0 = time.time()
    ttft = None
    n = 0
    finish = None
    prompt_tokens = None
    with urllib.request.urlopen(
        urllib.request.Request(
            f"http://127.0.0.1:{port}/v1/completions",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        ),
        timeout=7200,
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
            ch = (ev.get("choices") or [{}])[0]
            if ch.get("text"):
                if ttft is None:
                    ttft = time.time() - t0
                n += 1
            if ch.get("finish_reason"):
                finish = ch["finish_reason"]
            if ev.get("usage") and ev["usage"].get("prompt_tokens"):
                prompt_tokens = ev["usage"]["prompt_tokens"]
    wall = time.time() - t0
    if prompt_tokens is None:
        try:
            probe = json.load(urllib.request.urlopen(urllib.request.Request(
                f"http://127.0.0.1:{port}/v1/completions",
                data=json.dumps({"model": model, "prompt": prompt, "max_tokens": 1,
                                 "temperature": 0.0}).encode(),
                headers={"Content-Type": "application/json"}), timeout=7200))
            prompt_tokens = (probe.get("usage") or {}).get("prompt_tokens")
        except Exception:
            pass
    decode = (wall - ttft) if ttft is not None else None
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": n,
        "finish_reason": finish,
        "ttft_s": round(ttft, 4) if ttft else None,
        "decode_s": round(decode, 4) if decode else None,
        "wall_s": round(wall, 4),
        "prefill_tok_s": round(prompt_tokens / ttft, 2) if prompt_tokens and ttft else None,
        "ms_per_output_token": round(decode / n * 1000, 3) if decode and n else None,
        "output_tok_s": round(n / decode, 3) if decode and n else None,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8002)
    ap.add_argument("--context", type=int, required=True)
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--tag", default="")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    base = f"http://127.0.0.1:{args.port}"
    model = json.load(urllib.request.urlopen(f"{base}/v1/models", timeout=30))["data"][0]["id"]

    before = scrape(args.port)
    if before.get("requests_running", 0) or before.get("requests_waiting", 0):
        print("ERROR: server not idle before the cell; interval would not be clean",
              file=sys.stderr)
        return 2

    r = run_request(args.port, model, build_prompt(args.context), args.max_tokens)

    after = scrape(args.port)
    if after.get("requests_running", 0) or after.get("requests_waiting", 0):
        print("ERROR: server not idle after the cell; interval would not be clean",
              file=sys.stderr)
        return 2

    drafts = after.get("drafts", 0) - before.get("drafts", 0)
    draft_tokens = after.get("draft_tokens", 0) - before.get("draft_tokens", 0)
    accepted = after.get("accepted", 0) - before.get("accepted", 0)
    preempt = after.get("preemptions", 0) - before.get("preemptions", 0)

    pp_before = before.get("accepted_per_pos") or []
    pp_after = after.get("accepted_per_pos") or []
    per_pos = None
    if drafts and len(pp_before) == len(pp_after) and pp_before:
        per_pos = [round((a - b) / drafts, 4) for a, b in zip(pp_after, pp_before)]

    ct = r["completion_tokens"]
    cell = dict(r)
    cell.update({
        "tag": args.tag,
        "context_target": args.context,
        "spec_drafts": int(drafts) if drafts else None,
        "spec_draft_tokens": int(draft_tokens) if draft_tokens else None,
        "spec_accepted_tokens": int(accepted) if accepted else None,
        "accepted_per_pass": round(accepted / drafts, 4) if drafts else None,
        "accepted_per_output_token": round(accepted / ct, 4) if accepted and ct else None,
        "per_position_acceptance": per_pos,
        "preemptions_delta": int(preempt),
        "interval_note": "counters sampled immediately before/after one C1 request",
    })
    print(json.dumps(cell))
    if args.out:
        with open(args.out, "w") as fh:
            json.dump(cell, fh, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
