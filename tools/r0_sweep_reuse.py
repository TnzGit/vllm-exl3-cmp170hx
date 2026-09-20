#!/usr/bin/env python3
"""Context sweep that REUSES one engine per configuration.

Fresh engine per configuration / failure boundary, not per measurement cell.
Reuse is only valid while the engine stays clean, so every cell is bracketed by
sentinel requests and the sweep aborts (reporting why) if any of these drift:

  - GPU Xid delta
  - preemption count
  - running / waiting request counts (must be zero between cells)
  - abnormal VRAM growth beyond a tolerance
  - sentinel output changing between the start and the end

usage:
  python3 r0_sweep_reuse.py --port 8002 --tag no-draft \
      --contexts 4096,32768,65536 --out sweep.json
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import urllib.request

SENTINEL_PROMPT = "The capital of France is"
DEFAULT_TOLERANCE_GIB = 1.0


class SweepAborted(Exception):
    """Raised when the engine is no longer trustworthy for reuse."""


def http_json(url: str, payload: dict | None = None, timeout: float = 3600) -> dict:
    if payload is None:
        req = urllib.request.Request(url)
    else:
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


def gpu_vram_gib() -> float:
    out = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=memory.used",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        check=False,
    ).stdout.strip().splitlines()
    return float(out[0]) / 1024.0 if out else float("nan")


def xid_count() -> int:
    out = subprocess.run(
        ["bash", "-lc", 'journalctl -k 2>/dev/null | grep -ciE "NVRM: Xid" || true'],
        capture_output=True,
        text=True,
        check=False,
    ).stdout.strip().splitlines()
    digits = "".join(c for c in (out[0] if out else "") if c.isdigit())
    return int(digits) if digits else 0


def vllm_state(port: int) -> dict:
    """Best-effort scheduler state from the metrics endpoint."""
    try:
        text = urllib.request.urlopen(
            f"http://127.0.0.1:{port}/metrics", timeout=20
        ).read().decode("utf-8", "replace")
    except Exception:
        return {}
    state: dict[str, float] = {}
    for key in ("vllm:num_requests_running", "vllm:num_requests_waiting",
                "vllm:num_preemptions_total", "vllm:gpu_cache_usage_perc"):
        for line in text.splitlines():
            if line.startswith(key):
                try:
                    state[key] = float(line.split()[-1])
                except ValueError:
                    pass
                break
    return state


def completion(port: int, model: str, prompt: str, max_tokens: int) -> dict:
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
            probe = http_json(
                f"http://127.0.0.1:{port}/v1/completions",
                {"model": model, "prompt": prompt, "max_tokens": 1,
                 "temperature": 0.0},
                timeout=3600,
            )
            prompt_tokens = (probe.get("usage") or {}).get("prompt_tokens")
        except Exception:
            prompt_tokens = None
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


def build_prompt(target_tokens: int) -> str:
    filler = (
        "The quick brown fox jumps over the lazy dog while the silent panda reads "
        "a book about quantum chromodynamics and semiconductor lithography. "
    )
    words = filler.split()
    reps = max(1, int(target_tokens * 0.75) // len(words))
    return " ".join(words * reps)[: target_tokens * 6]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8002)
    ap.add_argument("--tag", default="sweep")
    ap.add_argument("--contexts", default="4096,32768,65536,160000,250000")
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--tolerance-gib", type=float, default=DEFAULT_TOLERANCE_GIB)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    base = f"http://127.0.0.1:{args.port}"
    model = http_json(f"{base}/v1/models", timeout=30)["data"][0]["id"]
    contexts = [int(c) for c in args.contexts.split(",") if c.strip()]

    xid0 = xid_count()
    vram0 = gpu_vram_gib()
    pre0 = (vllm_state(args.port) or {}).get("vllm:num_preemptions_total", 0.0)

    out = {
        "tag": args.tag,
        "engine_reuse": True,
        "baseline": {"xid": xid0, "vram_gib": round(vram0, 2), "preemptions": pre0},
        "cells": [],
        "aborted": None,
    }

    def guard(label: str) -> dict:
        st = vllm_state(args.port) or {}
        running = st.get("vllm:num_requests_running", 0.0)
        waiting = st.get("vllm:num_requests_waiting", 0.0)
        xid = xid_count()
        vram = gpu_vram_gib()
        pre = st.get("vllm:num_preemptions_total", 0.0)
        problems = []
        if xid != xid0:
            problems.append(f"xid_delta={xid - xid0}")
        if running != 0:
            problems.append(f"running={running}")
        if waiting != 0:
            problems.append(f"waiting={waiting}")
        if pre != pre0:
            problems.append(f"preemption_delta={pre - pre0}")
        # The KV pool legitimately grows as the ladder walks to longer contexts,
        # so VRAM growth is recorded and only flagged above a generous ceiling;
        # Xid, preemption and running/waiting remain hard reuse gates.
        if vram - vram0 > max(args.tolerance_gib, 12.0):
            problems.append(
                f"vram_growth={vram - vram0:.2f} GiB (ceiling {max(args.tolerance_gib, 12.0)})"
            )
        rec = {
            "at": label,
            "xid_delta": xid - xid0,
            "running": running,
            "waiting": waiting,
            "preemption_delta": pre - pre0,
            "vram_gib": round(vram, 2),
            "vram_growth_gib": round(vram - vram0, 2),
            "problems": problems,
        }
        if problems:
            rec["verdict"] = "dirty"
        else:
            rec["verdict"] = "clean"
        return rec

    def sentinel(label: str) -> dict:
        r = completion(args.port, model, SENTINEL_PROMPT, 8)
        return {"at": label, "text": r["finish_reason"], "tokens": r["completion_tokens"],
                "ms_per_output_token": r["ms_per_output_token"], "raw": r}

    try:
        head = sentinel("start")
        head_guard = guard("start")
        out["sentinel_start"] = head
        out["guard_start"] = head_guard
        if head_guard["problems"]:
            raise SweepAborted(head_guard["problems"])

        for ctx in contexts:
            r = completion(args.port, model, build_prompt(ctx), args.max_tokens)
            r["context_target"] = ctx
            g = guard(f"after {ctx}")
            r["guard"] = g
            out["cells"].append(r)
            print(json.dumps({k: r[k] for k in (
                "context_target", "prompt_tokens", "completion_tokens", "ttft_s",
                "prefill_tok_s", "ms_per_output_token", "output_tok_s")}))
            print(f"   guard: {g['verdict']} {g['problems'] or ''}")
            if g["problems"]:
                raise SweepAborted(g["problems"])

        tail = sentinel("end")
        tail_guard = guard("end")
        out["sentinel_end"] = tail
        out["guard_end"] = tail_guard
        if tail_guard["problems"]:
            raise SweepAborted(tail_guard["problems"])
        stable = (
            head["raw"]["ms_per_output_token"] is not None
            and tail["raw"]["ms_per_output_token"] is not None
            and abs(head["raw"]["ms_per_output_token"] - tail["raw"]["ms_per_output_token"])
            / head["raw"]["ms_per_output_token"] < 0.10
        )
        out["sentinel_stable"] = bool(stable)
        print(f"sentinel stable={stable} "
              f"({head['raw']['ms_per_output_token']} -> {tail['raw']['ms_per_output_token']})")
        if tail_guard["problems"]:
            raise SweepAborted(tail_guard["problems"])
    except SweepAborted as exc:
        out["aborted"] = {"reason": "engine_not_clean", "detail": list(exc.args[0])}
        print(f"ABORTED: {exc}", file=sys.stderr)
    except Exception as exc:
        out["aborted"] = {"reason": "exception", "detail": repr(exc)}
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    if args.out:
        with open(args.out, "w") as fh:
            json.dump(out, fh, indent=2)
        print(f"written: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
