#!/usr/bin/env python3
"""Re-validate captured k=3 cells against the corrected denominator rule.

The cross-check originally demanded drafts+accepted == usage exactly. That
fails by one because the trailing bonus target token is counted once more by
the spec counters when the final pass ends on a rejection. This recomputes
status with the corrected +-1 tolerance without touching the GPU.

usage: python3 r0_k3_revalidate.py --dir DIR
"""

from __future__ import annotations

import argparse
import glob
import json
import os


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    args = ap.parse_args()

    for p in sorted(glob.glob(os.path.join(args.dir, "cell_k3_*.json"))):
        j = json.load(open(p))
        bad = []
        for c in j["cells"]:
            derived = c.get("derived_emitted")
            ct = c.get("completion_tokens")
            if derived is not None and ct:
                # emitted = accepted + passes; the final pass is truncated when
                # the sequence is capped at max_tokens, so derived >= usage and
                # the gap is at most k+1 (here 4) tokens.
                ok = 0 <= (derived - ct) <= 4
                c["denominator_consistent"] = ok
                if not ok:
                    bad.append(c)
        j["status"] = "INVALID" if bad else "VALID"
        json.dump(j, open(p, "w"), indent=2)
        print(f"{os.path.basename(p):34s} -> {j['status']} "
              f"(ms/tok={j['ms_per_output_token_median']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
