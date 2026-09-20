#!/usr/bin/env python3
"""Resolve kernel callers and input shapes from a shapes-enabled torch trace.

torch records `Input Dims` on cpu_op events and a `correlation` id on kernel
events; the two are linked through the cpu_op's `External id`. This maps each
GPU kernel back to its launching CPU op and the shapes it was given, so a
kernel family like cuBLAS `gemv2T` can be attributed to a module instead of
guessed from its name.

usage: python3 r0_shapes_report.py TRACE.json.gz [--kernel SUBSTR] [--top 20]
"""

from __future__ import annotations

import argparse
import collections
import gzip
import json
import re


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("trace")
    ap.add_argument("--kernel", default=None)
    ap.add_argument("--op", default=None, help="filter on the CPU op name")
    ap.add_argument("--top", type=int, default=20)
    args = ap.parse_args()

    with gzip.open(args.trace) as fh:
        data = json.load(fh)
    events = data.get("traceEvents", [])

    by_ext: dict = {}
    for e in events:
        if e.get("cat") == "cpu_op":
            x = e.get("args", {}).get("External id")
            if x is not None:
                by_ext[x] = e

    agg: dict[tuple, list] = collections.defaultdict(lambda: [0, 0.0])
    total_kernels = 0
    matched = 0
    for e in events:
        if e.get("cat") != "kernel":
            continue
        total_kernels += 1
        name = e.get("name", "")
        if args.kernel and args.kernel not in name:
            continue
        c = e.get("args", {}).get("correlation")
        op = by_ext.get(c)
        opname = op.get("name", "?") if op else "<unmatched>"
        if op:
            matched += 1
        if args.op and args.op not in opname:
            continue
        dims = ""
        types = ""
        if op:
            dims = op.get("args", {}).get("Input Dims", "")
            types = op.get("args", {}).get("Input type", "")
            dims = str(dims)[:220]
            types = str(types)[:160]
        short = re.sub(r"<.*", "", name)[:58]
        agg[(short, opname[:44], dims, types)][0] += 1
        agg[(short, opname[:44], dims, types)][1] += e.get("dur", 0) / 1000.0

    print(f"kernel_events={total_kernels} matched_to_cpu_op={matched} "
          f"({100 * matched / max(total_kernels, 1):.1f}%)")
    print(f"{'calls':>7s} {'ms':>9s}  kernel | cpu_op | dims | types")
    for (short, opname, dims, types), (calls, ms) in sorted(
        agg.items(), key=lambda x: -x[1][1]
    )[: args.top]:
        print(f"{calls:7d} {ms:9.2f}  {short}")
        print(f"          cpu_op={opname}")
        if dims:
            print(f"          dims={dims}")
        if types:
            print(f"          types={types}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
