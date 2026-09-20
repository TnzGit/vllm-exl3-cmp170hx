#!/usr/bin/env python3
"""Summarize MTP k=3 qualification results and parity."""

from __future__ import annotations

import argparse
import glob
import json
import os


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    args = ap.parse_args()
    d = args.dir

    print("=== no-draft references ===")
    for ctx in (4096, 160000, 240000):
        p = os.path.join(d, f"ref_{ctx}.json")
        if not os.path.isfile(p):
            continue
        j = json.load(open(p))
        c = j["cells"][0]
        tp = c.get("token_pieces") or []
        print(f"  ctx={ctx:7d} ptok={j['prompt_tokens']:7d} usage_ctok={c['completion_tokens']}"
              f" pieces={len(tp):4d} ms/tok={j['ms_per_output_token_median']} status={j['status']}")

    print()
    print("=== k=3 long-context sweep ===")
    rows = []
    for p in sorted(glob.glob(os.path.join(d, "cell_k3_*.json"))):
        j = json.load(open(p))
        rows.append(j)
    order = {"k3-sentinel-in": 0, "k3-ctx32000": 1, "k3-ctx65000": 2,
             "k3-ctx126000": 3, "k3-ctx160000": 4, "k3-ctx200000": 5,
             "k3-ctx240000": 6, "k3-ctx250000": 7, "k3-sentinel-out": 9}
    rows.sort(key=lambda j: order.get(j["tag"], 8))
    hdr = (f"{'tag':18s} {'ptok':>8s} {'ms/tok':>8s} {'tok/s':>8s} "
           f"{'acc/pass':>9s} {'emit/pass':>10s} {'acc%':>6s} {'passes':>7s} {'status':>8s}")
    print(hdr)
    for j in rows:
        acc = j.get("accepted_per_pass")
        dpp = j.get("draft_tokens_per_pass") or 0
        accpct = (acc / dpp * 100) if (acc is not None and dpp) else None
        print(f"{j['tag']:18s} {j['prompt_tokens']:8d} "
              f"{str(j['ms_per_output_token_median']):>8s} "
              f"{str(j['output_tok_s_median']):>8s} "
              f"{str(acc):>9s} {str(j.get('emitted_per_pass')):>10s} "
              f"{(f'{accpct:.1f}' if accpct else '-'):>6s} "
              f"{str(j.get('verification_passes')):>7s} {j['status']:>8s}")

    print()
    print("=== greedy token-ID parity (no-draft vs k=3) ===")
    for ctx in (4096, 160000, 240000):
        p = os.path.join(d, f"parity_k3_{ctx}.json")
        if not os.path.isfile(p):
            print(f"  ctx={ctx}: no parity file")
            continue
        j = json.load(open(p))
        par = j.get("parity", {})
        print(f"  ctx={ctx:7d} status={par.get('status')} "
              f"first_mismatch={par.get('first_mismatch_index')} "
              f"ref_len={par.get('reference_len')} mtp_len={par.get('mtp_len')}")
        if par.get("status") == "MISMATCH":
            print(f"      ref_token={par.get('reference_token')} "
                  f"mtp_token={par.get('mtp_token')}")
            print(f"      ref_ctx={par.get('reference_context')}")
            print(f"      mtp_ctx={par.get('mtp_context')}")

    print()
    print("=== sentinel drift ===")
    si = os.path.join(d, "cell_k3_sentinel_in.json")
    so = os.path.join(d, "cell_k3_sentinel_out.json")
    if os.path.isfile(si) and os.path.isfile(so):
        a = json.load(open(si))["ms_per_output_token_median"]
        b = json.load(open(so))["ms_per_output_token_median"]
        drift = (b - a) / a * 100 if a else None
        print(f"  4K in={a} out={b} drift={drift:.2f}% "
              f"({'HEALTHY' if drift is not None and abs(drift) <= 3 else 'CHECK'})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
