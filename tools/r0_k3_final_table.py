#!/usr/bin/env python3
"""Render the final k=3 production-qualification table."""
from __future__ import annotations
import argparse, glob, json

ORDER = {"k3-sentinel-in":0,"k3-ctx32000":1,"k3-ctx65000":2,"k3-ctx126000":3,
         "k3-ctx160000":4,"k3-ctx200000":5,"k3-ctx240000":6,"k3-sentinel-out":7,
         "k3-recheck-4k":8}
LABELS = {"k3-sentinel-in":"4K in","k3-ctx32000":"32K","k3-ctx65000":"65K",
          "k3-ctx126000":"126K","k3-ctx160000":"160K","k3-ctx200000":"200K",
          "k3-ctx240000":"240K","k3-sentinel-out":"4K out","k3-recheck-4k":"4K recheck"}

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    args = ap.parse_args()
    rows = [json.load(open(p)) for p in glob.glob(args.dir + "/cell_k3_*.json")]
    rows.sort(key=lambda j: ORDER.get(j["tag"], 99))
    print("context      ptok    ms/tok    tok/s   acc/pass  emit/pass   acc%  status")
    for j in rows:
        acc = j["accepted_per_pass"]; dpp = j["draft_tokens_per_pass"] or 0
        pct = f"{acc/dpp*100:.1f}" if (acc and dpp) else "-"
        print(f"{LABELS.get(j['tag'], j['tag']):>9s} {j['prompt_tokens']:8d} "
              f"{j['ms_per_output_token_median']:9.3f} {j['output_tok_s_median']:8.3f} "
              f"{acc:9.4f} {j['emitted_per_pass']:10.4f} {pct:>6s} {j['status']:>7s}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
