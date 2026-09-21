#!/usr/bin/env python3
"""Ceiling math for the k=3 production Amdahl.

Compares each component's share against the measured profiler-OFF production
baseline so we get realistic end-to-end ceilings rather than "% of GPU time"
treated as automatically removable.

usage: python3 r0_k3_ceiling.py --dir DIR --production-ms 10.3
"""

from __future__ import annotations

import argparse
import json
import os


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--production-ms", type=float, default=10.3,
                    help="measured profiler-OFF production ms/output-token (4K)")
    args = ap.parse_args()
    d = args.dir

    for tag, ctx in (("4k", "4K"), ("160k", "160K")):
        p = os.path.join(d, f"amdahl_{tag}.json")
        if not os.path.isfile(p):
            continue
        j = json.load(open(p))
        tot = j["total_ms"]
        b = j["buckets"]
        print(f"=== {ctx} (passes={j['passes']:.0f} emitted={j['emitted']} "
              f"total_gpu_ms={tot:.2f}) ===")
        print("component            ms/pass   ms/token    share")
        for k, v in sorted(b.items(), key=lambda x: -x[1]["ms"]):
            print(f"{k:20s} {v['ms']/j['passes']:8.3f} "
                  f"{v['ms']/j['emitted']:9.4f} {100*v['ms']/tot:6.1f}%")

        moe = b["coop_moe_a"]["ms"] + b["coop_moe_b"]["ms"]
        lm = b["lm_head_candidate"]["ms"]
        print()
        print(f"MoE a+b   {moe:8.2f} ms/win {moe/j['passes']:7.3f} ms/pass "
              f"{moe/j['emitted']:8.4f} ms/tok {100*moe/tot:5.1f}%")
        print(f"lm_head   {lm:8.2f} ms/win {lm/j['passes']:7.3f} ms/pass "
              f"{lm/j['emitted']:8.4f} ms/tok {100*lm/tot:5.1f}%")
        print(f"MoE / lm_head ratio = {moe/lm:.2f}x")
        print()

    # Ceilings use the 4K window and the production profiler-OFF number.
    p = os.path.join(d, "amdahl_4k.json")
    j = json.load(open(p))
    tot, b = j["total_ms"], j["buckets"]
    print(f"=== ceilings vs production {args.production_ms} ms/output-token (4K) ===")
    print("component        share   full-removal ms/tok   full%   50%-fix ms/tok   50%")
    names = [("MoE a+b", b["coop_moe_a"]["ms"] + b["coop_moe_b"]["ms"]),
             ("lm_head", b["lm_head_candidate"]["ms"]),
             ("dense_gemm", b["dense_gemm"]["ms"]),
             ("dense_gemv", b["dense_gemv"]["ms"]),
             ("bf16_gemm", b["bf16_gemm"]["ms"]),
             ("dtype_copy", b["dtype_copy"]["ms"]),
             ("elementwise", b["elementwise"]["ms"]),
             ("hyper_conn", b["hyper_conn"]["ms"]),
             ("qsa", b["qsa"]["ms"]),
             ("fill_zero", b["fill_zero"]["ms"]),
             ("OTHER", b["OTHER"]["ms"])]
    for nm, ms in sorted(names, key=lambda x: -x[1]):
        sh = ms / tot
        print(f"{nm:16s} {sh*100:5.1f}%  {args.production_ms*sh:8.2f}        "
              f"{sh*100:5.1f}%  {args.production_ms*sh*0.5:8.2f}       {sh*50:5.1f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
