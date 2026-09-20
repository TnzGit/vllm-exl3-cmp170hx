#!/usr/bin/env python3
"""Post-COOP decode Amdahl table in ms/decode-step.

Buckets GPU kernels by name family and normalises by decode steps. Families are
matched on kernel-name substrings; anything unmatched is reported as OTHER and
never force-classified.

usage: python3 r0_coop_amdahl.py TRACE.json.gz --steps 31
"""

from __future__ import annotations

import argparse
import collections
import gzip
import json

BUCKETS = {
    "dense EXL3": ("exl3_gemv", "exl3_gemm"),
    "coop MoE": ("exl3_moe_coop",),
    "BF16 GEMM/GEMV": ("gemv2T_kernel", "gemvx::kernel", "cutlass_80_wmma"),
    "HyperConnection": ("_hc_",),
    "QSA/indexer": ("qsa", "sparse_attn_indexer", "persistent_topk"),
    "GDN/recurrent": ("gated_delta_rule", "causal_conv1d"),
    "PLE/n-gram": ("ngram",),
    "framework elem/copy": (
        "direct_copy", "bfloat16_copy", "float16_copy", "memcpy32_post",
        "FillFunctor", "compare_scalar", "index_elementwise", "scatter_gather",
        "CatArrayBatchedCopy", "splitKreduce", "arange", "masked_fill",
        "sigmoid", "layer_norm_fwd", "where_kernel", "bitonicSortKVInPlace",
    ),
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("trace")
    ap.add_argument("--steps", type=float, required=True)
    args = ap.parse_args()

    with gzip.open(args.trace) as fh:
        data = json.load(fh)

    agg: dict[str, list] = collections.defaultdict(lambda: [0, 0.0])
    total = 0.0
    for e in data.get("traceEvents", []):
        if e.get("cat") != "kernel":
            continue
        name = e.get("name", "")
        dur = e.get("dur", 0) / 1000.0
        total += dur
        hit = None
        for bucket, pats in BUCKETS.items():
            if any(p in name for p in pats):
                hit = bucket
                break
        key = hit or "OTHER"
        agg[key][0] += 1
        agg[key][1] += dur

    print(f"total_gpu_kernel_ms={total:.1f} steps={args.steps}")
    print(f"{'bucket':20s} {'calls':>8s} {'ms':>9s} {'pct':>7s} {'ms/step':>9s}")
    for key, (calls, ms) in sorted(agg.items(), key=lambda x: -x[1][1]):
        print(f"{key:20s} {calls:8d} {ms:9.2f} {100 * ms / total:6.2f}% {ms / args.steps:9.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
