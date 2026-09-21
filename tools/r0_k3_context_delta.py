#!/usr/bin/env python3
"""Decompose the 4K -> 160K latency difference into acceptance effect vs
per-pass kernel cost growth.

ms/output-token = ms/pass / emitted_per_pass, so a context delta can come from
(a) fewer emitted tokens per pass (acceptance) or (b) a genuinely more expensive
pass. This computes both contributions from the two captured windows.
"""

from __future__ import annotations

import argparse
import json
import os


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    args = ap.parse_args()
    d = args.dir

    caps = {}
    for ctx, tag in ((4096, "4k"), (160000, "160k")):
        p = os.path.join(d, f"cap_{tag}.json")
        if os.path.isfile(p):
            caps[ctx] = json.load(open(p))

    print("=== window counters ===")
    for ctx, c in sorted(caps.items()):
        a = c["accepted_in_window"]
        p = c["verification_passes_in_window"]
        e = a + p
        print(f"ctx={ctx:7d} passes={p} draft_tokens={c['draft_tokens_in_window']} "
              f"accepted={a} emitted={e} emitted/pass={e/p:.3f} "
              f"acc_rate={a/c['draft_tokens_in_window']*100:.1f}%")

    amalg = {}
    for ctx, tag in ((4096, "4k"), (160000, "160k")):
        p = os.path.join(d, f"amdahl_{tag}.json")
        if os.path.isfile(p):
            amalg[ctx] = json.load(open(p))

    print()
    print("=== decomposition ===")
    a4, a16 = caps[4096], caps[160000]
    e4 = a4["accepted_in_window"] + a4["verification_passes_in_window"]
    e16 = a16["accepted_in_window"] + a16["verification_passes_in_window"]
    ms4 = amalg[4096]["total_ms"]
    ms16 = amalg[160000]["total_ms"]
    pass4 = ms4 / a4["verification_passes_in_window"]
    pass16 = ms16 / a16["verification_passes_in_window"]
    t4 = pass4 / e4 * 14
    t16 = pass16 / e16 * 14

    print(f"4K   ms/pass={pass4:.3f}  emitted/pass={e4/a4['verification_passes_in_window']:.3f}")
    print(f"160K ms/pass={pass16:.3f}  emitted/pass={e16/a16['verification_passes_in_window']:.3f}")
    print()
    # ms/output-token scaled to the whole window: ms/pass / emitted_per_pass
    tps4 = pass4 / (e4 / 14)
    tps16 = pass16 / (e16 / 14)
    print(f"4K   GPU ms/output-token={tps4:.4f}")
    print(f"160K GPU ms/output-token={tps16:.4f}")
    print(f"delta={(tps16 - tps4):+.4f} ({(tps16/tps4 - 1)*100:+.1f}%)")

    # counterfactual: 160K passes at 4K emitted/pass isolates the acceptance effect
    cf_acc = pass4 / (e16 / 14)
    print()
    print(f"counterfactual: 4K pass cost, 160K acceptance -> {cf_acc:.4f} ms/token")
    print(f"  acceptance effect = {cf_acc - tps4:+.4f} ms/token "
          f"({(cf_acc/tps4 - 1)*100:+.1f}%)")
    print(f"  per-pass cost effect = {tps16 - cf_acc:+.4f} ms/token "
          f"({(tps16/cf_acc - 1)*100:+.1f}%)")
    tot = (cf_acc - tps4) + (tps16 - cf_acc)
    print(f"  sum={tot:+.4f} (check delta={tps16 - tps4:+.4f})")
    if abs(tot - (tps16 - tps4)) < 1e-6:
        print("  decomposition closes exactly")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
