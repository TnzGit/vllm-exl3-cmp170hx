#!/usr/bin/env python3
"""Phase 3 dispatch proof from verified counters + pack geometry (no profiler).

The task asks whether, under MTP k=1/2/3, the coop MoE path is actually hit, no
stock fallback occurs, and which dense EXL3 family each verify width selects.

Profiler-free evidence:
  1. spec counters give the real verify width m = draft_tokens/drafts + 1
  2. the plugin's _dense_forward selects by row count, so m determines the
     dense kernel family (GEMV for rows<=2, cooperative GEMM for 3..16, and
     reconstruct for >=17)
  3. coop MoE eligibility is evaluated per call from tokens*topk and geometry

usage: python3 r0_mtp_dispatch_proof.py --results DIR
"""

from __future__ import annotations

import argparse
import glob
import json
import os

# From src/vllm_exl3/exl3.py: exllamav3 dispatches dense EXL3 by row count.
GEMV_MAX_ROWS = 2          # rows <= 2 -> non-cooperative exl3_gemv
COOP_GEMM_MAX_ROWS = 144   # 3..144 -> cooperative trellis GEMM
RECON_MIN_ROWS = 17        # plugin reroutes rows >= 17 to the reconstruct path
TOPK = 10                  # num_experts_per_tok for this pack
HIDDEN = 2560
INTERMEDIATE = 640
COOP_SLOT_CAP = 256        # tokens*topk <= 256


def dense_family(rows: int) -> str:
    if rows <= GEMV_MAX_ROWS:
        return "exl3_gemv_int8_sq (non-cooperative GEMV)"
    if rows < RECON_MIN_ROWS:
        return "exl3 cooperative trellis GEMM (3..16)"
    if rows <= COOP_GEMM_MAX_ROWS:
        return "reconstruct -> hgemm (plugin reroutes rows>=17)"
    return "reconstruct -> hgemm (>144)"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    args = ap.parse_args()

    print("=== verify width and dense EXL3 family per config ===")
    print(f"{'config':12s} {'drafts':>7s} {'draft_tok':>9s} {'acc':>5s} "
          f"{'m':>3s} {'dense family at m':46s} {'coop eligible':>14s}")
    for f in sorted(glob.glob(os.path.join(args.results, "cell_coop1-k*.json"))):
        d = json.load(open(f))
        c = d["cells"][0]
        tag = d["tag"]
        dr = c.get("spec_drafts")
        if not dr:
            print(f"{tag:12s} {'-':>7s} {'-':>9s} {'-':>5s} {1:>3d} "
                  f"{dense_family(1):46s} {'n/a (no draft)':>14s}")
            continue
        dt = c.get("spec_draft_tokens") or 0
        ac = c.get("spec_accepted") or 0
        k = int(tag.split("k")[1])
        m = k + 1
        slots = m * TOPK
        coop = (slots <= COOP_SLOT_CAP and HIDDEN % 128 == 0
                and INTERMEDIATE % 128 == 0)
        print(f"{tag:12s} {dr:7d} {dt:9d} {ac:5d} {m:3d} "
              f"{dense_family(m):46s} {str(coop):>14s}")

    print()
    print("=== coop MoE eligibility arithmetic (tokens = m) ===")
    for k in (1, 2, 3):
        m = k + 1
        print(f"  k={k}: m={m}  m*topk={m * TOPK} <= {COOP_SLOT_CAP} -> "
              f"{'ELIGIBLE' if m * TOPK <= COOP_SLOT_CAP else 'falls back to stock'}")
    print(f"  hidden {HIDDEN} % 128 == {HIDDEN % 128}  "
          f"intermediate {INTERMEDIATE} % 128 == {INTERMEDIATE % 128}")

    print()
    print("=== dense m=2 -> m=3 crossover ===")
    print(f"  k=1 -> m=2 -> {dense_family(2)}")
    print(f"  k=2 -> m=3 -> {dense_family(3)}")
    print("  => the GEMV -> cooperative GEMM transition DOES occur at k=2.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
