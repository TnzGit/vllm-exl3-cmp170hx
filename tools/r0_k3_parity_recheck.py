#!/usr/bin/env python3
"""Recompute full greedy token-ID parity from captured files (no GPU needed).

The parity step originally reported NO_REFERENCE_TOKENS because the loader read
token_pieces from the top level while the cell stores it inside cells[0]. This
recomputes the comparison from the already-captured k=3 parity cells and the
no-draft references.

usage: python3 r0_k3_parity_recheck.py --dir DIR
"""

from __future__ import annotations

import argparse
import glob
import json
import os


def tokens_of(path: str) -> list[str]:
    """Flatten token pieces.

    A stream chunk's logprobs.tokens can itself be a list of several tokens
    (that is exactly the speculative multi-token chunk), and the reference path
    yields flat single-token strings. Flatten both so the comparison is
    token-vs-token rather than token-vs-chunk.
    """
    j = json.load(open(path))
    tp = j.get("token_pieces")
    if not tp:
        tp = (j.get("cells") or [{}])[0].get("token_pieces") or []
    flat: list[str] = []
    for t in tp:
        if isinstance(t, list):
            flat.extend(t)
        else:
            flat.append(t)
    return flat


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    args = ap.parse_args()
    d = args.dir

    print(f"{'ctx':>8s} {'status':>10s} {'ref_len':>8s} {'mtp_len':>8s} "
          f"{'first_mismatch':>15s}")
    for p in sorted(glob.glob(os.path.join(d, "parity_k3_*.json"))):
        ctx = os.path.basename(p).replace("parity_k3_", "").replace(".json", "")
        ref_path = os.path.join(d, f"ref_{ctx}.json")
        if not os.path.isfile(ref_path):
            print(f"{ctx:>8s} {'NO_REF':>10s}")
            continue
        ref = tokens_of(ref_path)
        got = tokens_of(p)
        if not ref or not got:
            print(f"{ctx:>8s} {'EMPTY':>10s} ref={len(ref)} mtp={len(got)}")
            continue
        n = min(len(ref), len(got))
        first = next((i for i in range(n) if ref[i] != got[i]), None)
        status = "PASS" if (first is None and len(ref) == len(got)) else "MISMATCH"
        print(f"{ctx:>8s} {status:>10s} {len(ref):8d} {len(got):8d} "
              f"{str(first):>15s}")
        if status == "MISMATCH":
            i = first if first is not None else min(len(ref), len(got))
            print(f"         ref_token={ref[i]!r} mtp_token={got[i]!r}")
            lo = max(0, i - 3)
            print(f"         ref_ctx={ref[lo:i + 4]}")
            print(f"         mtp_ctx={got[lo:i + 4]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
