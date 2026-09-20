#!/usr/bin/env python3
"""Per-decode-step EXL3 kernel breakdown from a vLLM torch trace.

Splits EXL3 GPU kernel time into dense GEMV/GEMM and cooperative-MoE parts and
normalises by decode steps, so the post-COOP Amdahl is expressed in
ms/decode-step rather than a profiler-length-dependent window total.

usage: python3 r0_exl3_steps.py TRACE.json.gz [--steps 30.35]
"""

from __future__ import annotations

import argparse
import collections
import gzip
import json


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("trace")
    ap.add_argument("--steps", type=float, required=True)
    args = ap.parse_args()

    with gzip.open(args.trace) as fh:
        data = json.load(fh)

    agg: dict[str, list] = collections.defaultdict(lambda: [0, 0.0])
    for e in data.get("traceEvents", []):
        if e.get("cat") != "kernel":
            continue
        name = e.get("name", "")
        if "exl3_gemv_int8_sq" in name:
            key = "dense gemv_int8_sq K=" + name.split("<")[1].split(",")[0]
        elif "exl3_gemv_kernel" in name:
            key = "dense gemv_kernel K=" + name.split("<")[1].split(",")[0]
        elif "exl3_gemm_kernel" in name:
            key = "dense gemm_kernel K=" + name.split("<")[1].split(",")[0]
        elif "exl3_moe_coop_a" in name:
            key = "coop moe a"
        elif "exl3_moe_coop_b" in name:
            key = "coop moe b"
        else:
            continue
        agg[key][0] += 1
        agg[key][1] += e.get("dur", 0) / 1000.0

    header = ("kernel", "calls", "ms", "ms/step", "calls/step")
    print(f"{header[0]:22s} {header[1]:>7s} {header[2]:>9s} {header[3]:>9s} {header[4]:>11s}")
    total = 0.0
    for key, (calls, ms) in sorted(agg.items(), key=lambda x: -x[1][1]):
        total += ms
        print(f"{key:22s} {calls:7d} {ms:9.2f} {ms / args.steps:9.3f} {calls / args.steps:11.2f}")
    print(f"{'EXL3 total':22s} {'':7s} {total:9.2f} {total / args.steps:9.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
