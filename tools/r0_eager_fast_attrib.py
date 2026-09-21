#!/usr/bin/env python3
"""Fast eager attribution for the post-COOP residual kernel families.

The stock analyzer walks full stacks for every event and does not finish in
reasonable time on a 1.6M-event eager trace. This one only needs what the
attribution question requires:

  kernel --(correlation == cpu_op."External id")--> CPU op
  CPU op --(time containment)--> nearest enclosing python_function

so it is O(n log n) with dicts and a sorted interval index over python frames.

Outputs target-kernel GPU time grouped by CPU op and by python parent, plus the
correlation coverage that decides whether caller attribution is trustworthy.

usage: python3 r0_eager_fast_attrib.py TRACE.json.gz [--top 15]
"""

from __future__ import annotations

import argparse
import bisect
import collections
import gzip
import json
import re

TARGET_PATTERNS = {
    "gemv2T": ("gemv2T_kernel",),
    "gemvx": ("gemvx::kernel",),
    "cutlass_wmma": ("cutlass_80_wmma",),
    "direct_copy": ("direct_copy_kernel",),
    "bfloat16_copy": ("bfloat16_copy_kernel",),
    "memcpy32_post": ("memcpy32_post",),
    "FillFunctor": ("FillFunctor",),
    "compare_scalar": ("compare_scalar",),
    "index_elementwise": ("index_elementwise_kernel",),
    "scatter_gather": ("_scatter_gather_elementwise",),
    "topk_sort": ("persistent_topk_kernel", "bitonicSortKVInPlace"),
    "coop_moe": ("exl3_moe_coop",),
    "dense_exl3": ("exl3_gemv", "exl3_gemm"),
}


def family_of(name: str) -> str | None:
    for fam, pats in TARGET_PATTERNS.items():
        if any(p in name for p in pats):
            return fam
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("trace")
    ap.add_argument("--top", type=int, default=12)
    args = ap.parse_args()

    with gzip.open(args.trace) as fh:
        events = json.load(fh).get("traceEvents", [])

    # --- indexes -----------------------------------------------------------
    # torch profiler chains cpu_op -> cuda_runtime -> kernel:
    #   cpu_op."External id" == cuda_runtime."External id"
    #   cuda_runtime."correlation" == kernel."correlation"
    # Matching kernel.correlation straight to cpu_op."External id" skips the
    # middle hop and pairs kernels with unrelated memory ops.
    by_extid: dict = {}
    runtime_by_corr: dict = {}
    py_frames: list[tuple[float, float, str]] = []
    for e in events:
        cat = e.get("cat")
        if cat == "cpu_op":
            x = e.get("args", {}).get("External id")
            if x is not None:
                by_extid[x] = e
        elif cat == "cuda_runtime":
            a = e.get("args", {})
            c = a.get("correlation")
            x = a.get("External id")
            if c is not None and x is not None:
                runtime_by_corr[c] = x
        elif cat == "python_function":
            nm = e.get("name", "")
            if nm:
                ts = e.get("ts", 0.0)
                py_frames.append((ts, ts + e.get("dur", 0.0), nm))
    py_frames.sort()
    py_starts = [f[0] for f in py_frames]

    def py_parent(ts: float) -> str:
        # nearest enclosing python frame: last frame starting at or before ts
        i = bisect.bisect_right(py_starts, ts) - 1
        while i >= 0:
            s, e, nm = py_frames[i]
            if s <= ts <= e:
                return nm
            i -= 1
        return "<no-python-parent>"

    # --- aggregate ---------------------------------------------------------
    total_ms = 0.0
    total_target_ms = 0.0
    matched_target_ms = 0.0
    per_family_ms: dict[str, float] = collections.defaultdict(float)
    per_family_count: dict[str, int] = collections.defaultdict(int)
    by_op: dict[tuple, list] = collections.defaultdict(lambda: [0, 0.0])
    by_parent: dict[tuple, list] = collections.defaultdict(lambda: [0, 0.0])
    coop_a = 0

    for e in events:
        if e.get("cat") != "kernel":
            continue
        name = e.get("name", "")
        dur = e.get("dur", 0) / 1000.0
        total_ms += dur
        if "exl3_moe_coop_a" in name:
            coop_a += 1
        fam = family_of(name)
        if fam is None:
            continue
        total_target_ms += dur
        per_family_ms[fam] += dur
        per_family_count[fam] += 1
        ext = runtime_by_corr.get(e.get("args", {}).get("correlation"))
        op = by_extid.get(ext) if ext is not None else None
        if op is None:
            by_op[(fam, "<unmatched>", "")][0] += 1
            by_op[(fam, "<unmatched>", "")][1] += dur
            by_parent[(fam, "<unmatched>")][0] += 1
            by_parent[(fam, "<unmatched>")][1] += dur
            continue
        matched_target_ms += dur
        opname = op.get("name", "?")
        dims = str(op.get("args", {}).get("Input Dims", ""))[:110]
        parent = py_parent(op.get("ts", 0.0))
        by_op[(fam, opname[:52], dims)][0] += 1
        by_op[(fam, opname[:52], dims)][1] += dur
        by_parent[(fam, parent[:70])][0] += 1
        by_parent[(fam, parent[:70])][1] += dur

    steps = coop_a / 48.0
    print(f"total_kernel_events_ms={total_ms:.1f}")
    print(f"coop_a_kernel_count={coop_a}")
    print(f"decode_steps={steps:.2f}")
    print(f"target_kernel_ms={total_target_ms:.1f}  "
          f"matched_ms={matched_target_ms:.1f}  "
          f"target_correlation_coverage="
          f"{100 * matched_target_ms / max(total_target_ms, 1e-9):.1f}%")
    print()
    print(f"{'family':18s} {'calls':>8s} {'ms':>9s} {'ms/step':>9s}")
    for fam, ms in sorted(per_family_ms.items(), key=lambda x: -x[1]):
        step = ms / steps if steps else float("nan")
        print(f"{fam:18s} {per_family_count[fam]:8d} {ms:9.2f} {step:9.4f}")

    print()
    print("=== matched GPU time by (family, CPU op) — unmatched excluded ===")
    matched = {k: v for k, v in by_op.items() if k[1] != "<unmatched>"}
    for (fam, op, dims), (calls, ms) in sorted(matched.items(), key=lambda x: -x[1][1])[: args.top]:
        print(f"{ms:9.2f} ms  {calls:7d}x  {fam:14s} {op}")
        if dims:
            print(f"                                  dims={dims}")
    print()
    print("=== per-family matched coverage ===")
    fam_total = collections.defaultdict(float)
    fam_match = collections.defaultdict(float)
    for (fam, op, _dims), (_c, ms) in by_op.items():
        fam_total[fam] += ms
        if op != "<unmatched>":
            fam_match[fam] += ms
    for fam in sorted(fam_total, key=lambda f: -fam_total[f]):
        pct = 100 * fam_match[fam] / max(fam_total[fam], 1e-9)
        print(f"{fam:18s} total={fam_total[fam]:8.2f} ms  matched={fam_match[fam]:8.2f} ms  {pct:5.1f}%")

    print()
    print("=== GPU time by (family, python parent) ===")
    for (fam, parent), (calls, ms) in sorted(by_parent.items(), key=lambda x: -x[1][1])[: args.top]:
        print(f"{ms:9.2f} ms  {calls:7d}x  {fam:14s} {parent}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
