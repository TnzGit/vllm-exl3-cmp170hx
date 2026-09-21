#!/usr/bin/env python3
"""Split k=3 production work into draft-side (m=1) vs verify-side (m=4).

Evidence sources, in order of confidence:
  1. input row count from the correlated CPU op's "Input Dims" — draft MTP steps
     run one row at a time (m=1); target verification runs m = k+1 = 4 rows.
  2. call-count arithmetic against 48 layers and the known passes.
  3. kernel family (GEMV for rows<=2, cooperative GEMM for 3..16).

Anything not backed by an observed row count is labelled INFERRED.

usage: python3 r0_k3_draft_verify.py TRACE.json.gz --passes 14 [--top 20]
"""

from __future__ import annotations

import argparse
import collections
import gzip
import json
import re


def row_of(dims: str) -> int | None:
    """First dimension of the first input, from '[[4, 2560], []]' -> 4."""
    m = re.match(r"\[\[(\d+)", dims or "")
    return int(m.group(1)) if m else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("trace")
    ap.add_argument("--passes", type=float, required=True)
    ap.add_argument("--top", type=int, default=20)
    args = ap.parse_args()

    with gzip.open(args.trace) as fh:
        events = json.load(fh).get("traceEvents", [])

    rt: dict = {}
    cpu: dict = {}
    for e in events:
        if e.get("cat") == "cuda_runtime":
            a = e.get("args", {})
            if a.get("correlation") is not None and a.get("External id") is not None:
                rt[a["correlation"]] = a["External id"]
        elif e.get("cat") == "cpu_op":
            x = e.get("args", {}).get("External id")
            if x is not None:
                cpu[x] = e

    # side -> kernel-short -> [calls, ms]
    agg: dict[tuple, list] = collections.defaultdict(lambda: [0, 0.0])
    unknown_ms = 0.0
    total_ms = 0.0

    for e in events:
        if e.get("cat") != "kernel":
            continue
        dur = e.get("dur", 0) / 1000.0
        total_ms += dur
        ext = rt.get(e.get("args", {}).get("correlation"))
        op = cpu.get(ext) if ext is not None else None
        if op is None:
            unknown_ms += dur
            continue
        dims = str(op.get("args", {}).get("Input Dims", ""))
        rows = row_of(dims)
        if rows is None:
            agg[("unknown-rows", op.get("name", "?")[:40])][0] += 1
            agg[("unknown-rows", op.get("name", "?")[:40])][1] += dur
            continue
        side = "VERIFY m=4" if rows == 4 else ("DRAFT m=1" if rows == 1
                                               else f"rows={rows}")
        nm = e.get("name", "")[:64]
        agg[(side, nm)][0] += 1
        agg[(side, nm)][1] += dur

    print(f"passes={args.passes} total_gpu_ms={total_ms:.2f} "
          f"unmatched(kernel w/o cpu op)={unknown_ms:.2f} "
          f"({100*unknown_ms/total_ms:.0f}%) — unmatched kernels are graph-replay "
          f"and cannot be attributed by caller")
    print()
    side_ms: dict[str, float] = collections.defaultdict(float)
    side_calls: dict[str, int] = collections.defaultdict(int)
    for (side, _nm), (c, ms) in agg.items():
        side_ms[side] += ms
        side_calls[side] += c
    print(f"{'side':14s} {'ms':>9s} {'calls':>7s} {'ms/pass':>9s} "
          f"{'calls/pass':>11s}")
    for side, ms in sorted(side_ms.items(), key=lambda x: -x[1]):
        print(f"{side:14s} {ms:9.2f} {side_calls[side]:7d} "
              f"{ms/args.passes:9.3f} {side_calls[side]/args.passes:11.2f}")

    print()
    print("=== top attributed (side, kernel) ===")
    for (side, nm), (c, ms) in sorted(agg.items(),
                                      key=lambda x: -x[1][1])[:args.top]:
        print(f"{ms:9.2f} ms {c:6d}x  {side:14s} {nm}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
