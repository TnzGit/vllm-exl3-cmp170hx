#!/usr/bin/env python3
"""Post-MTP k=3 production Amdahl analyzer.

Two jobs:

1. Kernel Amdahl from a production CUDA-graph trace, normalised per
   verification pass and per authoritative emitted token (API
   usage.completion_tokens — never chunk counts).

2. Caller attribution via the CORRECT two-hop chain:
     cpu_op.External id == cuda_runtime.External id
     cuda_runtime.correlation == kernel.correlation
   The earlier mistake was matching kernel.correlation straight to
   cpu_op.External id, which produced a uniform bogus coverage and paired
   kernels with unrelated memory ops.

usage:
  python3 r0_k3_amdahl.py TRACE.json.gz --passes N --emitted N [--top 25]
"""

from __future__ import annotations

import argparse
import bisect
import collections
import gzip
import json
import re

# Kernel-name classification. Anything unmatched stays OTHER and is never
# force-classified.
CLASSIFIERS = [
    ("draft_lm_head", ("exl3_gemv_int8_sq_kernel<5",)),          # resolved below
    ("coop_moe_a", ("exl3_moe_coop_a",)),
    ("coop_moe_b", ("exl3_moe_coop_b",)),
    ("stock_moe", ("exl3_moe_kernel",)),
    ("dense_gemv", ("exl3_gemv_int8_sq_kernel", "exl3_gemv_kernel")),
    ("dense_gemm", ("exl3_gemm_kernel",)),
    ("bf16_gemm", ("gemv2T_kernel", "gemvx::kernel", "cutlass_80_wmma",
                   "splitKreduce")),
    ("hyper_conn", ("_hc_",)),
    ("qsa", ("qsa", "sparse_attn_indexer", "_expand_qsa_indices",
             "_qsa_mqa_paged")),
    ("gdn", ("gated_delta_rule", "causal_conv1d")),
    ("ple_ngram", ("ngram",)),
    ("topk_sort", ("persistent_topk", "bitonicSort", "radix_sort")),
    ("index_scatter", ("index_elementwise", "scatter", "gather",
                       "indexSelectSmallIndex")),
    ("fill_zero", ("FillFunctor", "zero_")),
    ("dtype_copy", ("direct_copy", "bfloat16_copy", "float16_copy",
                    "memcpy32_post", "CatArrayBatchedCopy")),
    ("elementwise", ("elementwise_kernel", "unrolled_elementwise",
                     "vectorized_elementwise", "sigmoid", "MulFunctor",
                     "layer_norm_fwd", "where_kernel", "compare_scalar",
                     "arange")),
    ("memcpy", ("Memcpy", "Memset", "memcpy", "memset")),
]


def classify(name: str) -> tuple[str, bool]:
    """Return (bucket, is_lm_head_candidate)."""
    for bucket, pats in CLASSIFIERS:
        if any(p in name for p in pats):
            lm = bucket == "draft_lm_head"
            return ("dense_gemv" if lm else bucket), lm
    return "OTHER", False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("trace")
    ap.add_argument("--passes", type=float, required=True,
                    help="verification passes in the profiled window")
    ap.add_argument("--emitted", type=int, required=True,
                    help="authoritative emitted tokens in the window")
    ap.add_argument("--top", type=int, default=25)
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()

    with gzip.open(args.trace) as fh:
        events = json.load(fh).get("traceEvents", [])

    # --- two-hop correlation index ----------------------------------------
    runtime_by_corr: dict = {}
    cpu_by_extid: dict = {}
    py_frames: list[tuple[float, float, str]] = []
    for e in events:
        cat = e.get("cat")
        if cat == "cuda_runtime":
            a = e.get("args", {})
            c, x = a.get("correlation"), a.get("External id")
            if c is not None and x is not None:
                runtime_by_corr[c] = x
        elif cat == "cpu_op":
            x = e.get("args", {}).get("External id")
            if x is not None:
                cpu_by_extid[x] = e
        elif cat == "python_function":
            nm = e.get("name", "")
            if nm:
                ts = e.get("ts", 0.0)
                py_frames.append((ts, ts + e.get("dur", 0.0), nm))
    py_frames.sort()
    py_starts = [f[0] for f in py_frames]

    def py_parent(ts: float) -> str:
        i = bisect.bisect_right(py_starts, ts) - 1
        while i >= 0:
            s, en, nm = py_frames[i]
            if s <= ts <= en:
                return nm
            i -= 1
        return "<no-python-parent>"

    # --- aggregate ---------------------------------------------------------
    total_ms = 0.0
    bucket_ms: dict[str, float] = collections.defaultdict(float)
    bucket_calls: dict[str, int] = collections.defaultdict(int)
    raw: dict[str, list] = collections.defaultdict(lambda: [0, 0.0, 0.0])
    # draft/verify probes using call-count arithmetic relative to passes
    probe = collections.Counter()
    probe_ms = collections.defaultdict(float)
    matched_ms = 0.0
    graph_kernels = 0
    kernels = 0

    for e in events:
        if e.get("cat") != "kernel":
            continue
        name = e.get("name", "")
        dur = e.get("dur", 0) / 1000.0
        kernels += 1
        total_ms += dur
        if e.get("args", {}).get("graph node id") is not None:
            graph_kernels += 1

        bucket, lm_candidate = classify(name)
        # lm_head is 2560 -> 248320; identify by the K5 gemv plus a large-N
        # signature captured separately via caller rows below.
        if lm_candidate:
            bucket = "lm_head_candidate"

        bucket_ms[bucket] += dur
        bucket_calls[bucket] += 1
        r = raw[name[:110]]
        r[0] += 1
        r[1] += dur

        ext = runtime_by_corr.get(e.get("args", {}).get("correlation"))
        op = cpu_by_extid.get(ext) if ext is not None else None
        if op is not None:
            matched_ms += dur
            r[2] += dur
            opname = op.get("name", "?")
            dims = str(op.get("args", {}).get("Input Dims", ""))[:90]
            probe[(bucket, opname[:46], dims)] += 1
            probe_ms[(bucket, opname[:46], dims)] += dur

    cov = 100 * matched_ms / max(total_ms, 1e-9)
    print(f"trace={args.trace}")
    print(f"kernels={kernels} graph_replayed={graph_kernels} "
          f"({100*graph_kernels/max(kernels,1):.0f}%)")
    print(f"passes={args.passes} emitted_tokens={args.emitted}")
    print(f"total_gpu_ms={total_ms:.2f}  caller_matched_ms={matched_ms:.2f}  "
          f"coverage={cov:.1f}%")
    print()

    print(f"{'bucket':22s} {'ms/win':>9s} {'%':>6s} {'calls':>7s} "
          f"{'calls/pass':>11s} {'ms/pass':>9s} {'ms/token':>9s}")
    for b, ms in sorted(bucket_ms.items(), key=lambda x: -x[1]):
        print(f"{b:22s} {ms:9.2f} {100*ms/total_ms:5.1f}% {bucket_calls[b]:7d} "
              f"{bucket_calls[b]/args.passes:11.2f} {ms/args.passes:9.3f} "
              f"{ms/args.emitted:9.4f}")

    print()
    print("=== top raw kernels ===")
    print(f"{'ms':>9s} {'calls':>7s} {'ms/pass':>9s}  kernel")
    for nm, (c, ms, mms) in sorted(raw.items(), key=lambda x: -x[1][1])[:args.top]:
        print(f"{ms:9.2f} {c:7d} {ms/args.passes:9.3f}  {nm}")

    print()
    print("=== top (bucket, CPU op, dims) — caller attribution ===")
    for (b, op, dims), ms in sorted(probe_ms.items(),
                                     key=lambda x: -x[1])[:args.top]:
        c = probe[(b, op, dims)]
        print(f"{ms:9.2f} ms {c:6d}x {b:18s} {op}"
              + (f"  dims={dims}" if dims else ""))

    if args.json_out:
        json.dump({
            "trace": args.trace, "kernels": kernels,
            "graph_replayed": graph_kernels, "passes": args.passes,
            "emitted": args.emitted, "total_ms": total_ms,
            "coverage_pct": cov,
            "buckets": {b: {"ms": bucket_ms[b], "calls": bucket_calls[b]}
                        for b in bucket_ms},
        }, open(args.json_out, "w"), indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
