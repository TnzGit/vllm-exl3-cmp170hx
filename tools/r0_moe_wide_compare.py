#!/usr/bin/env python3
"""Compare wide/wide vs narrow/narrow EXL3 cooperative MoE traces.

The two traces must be analyzed with r0_k3_amdahl.py first.  This report keeps
stage A and B separate so a global narrow regression cannot hide a useful
B-only win (or vice versa).

usage:
  python3 r0_moe_wide_compare.py wide.json narrow.json
"""

from __future__ import annotations

import argparse
import json


BUCKETS = ("coop_moe_a", "coop_moe_b")


def per_pass(j: dict, bucket: str) -> tuple[float, float]:
    d = j["buckets"].get(bucket, {"ms": 0.0, "calls": 0})
    passes = float(j["passes"])
    return float(d["ms"]) / passes, float(d["calls"]) / passes


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("wide_json")
    ap.add_argument("narrow_json")
    args = ap.parse_args()

    wide = json.load(open(args.wide_json))
    narrow = json.load(open(args.narrow_json))

    out: dict[str, object] = {}
    wide_total = 0.0
    narrow_total = 0.0
    ideal_total = 0.0

    for bucket in BUCKETS:
        w_ms, w_calls = per_pass(wide, bucket)
        n_ms, n_calls = per_pass(narrow, bucket)
        best = min(w_ms, n_ms)
        out[bucket] = {
            "wide_ms_per_pass": round(w_ms, 6),
            "narrow_ms_per_pass": round(n_ms, 6),
            "narrow_minus_wide_ms": round(n_ms - w_ms, 6),
            "narrow_vs_wide_pct": round((n_ms / w_ms - 1.0) * 100.0, 3)
            if w_ms else None,
            "wide_calls_per_pass": round(w_calls, 4),
            "narrow_calls_per_pass": round(n_calls, 4),
            "best_geometry": "narrow" if n_ms < w_ms else "wide",
            "best_ms_per_pass": round(best, 6),
        }
        wide_total += w_ms
        narrow_total += n_ms
        ideal_total += best

    emitted_per_pass = float(wide["emitted"]) / float(wide["passes"])
    ideal_saving = wide_total - ideal_total
    out["summary"] = {
        "wide_wide_ms_per_pass": round(wide_total, 6),
        "narrow_narrow_ms_per_pass": round(narrow_total, 6),
        "narrow_narrow_delta_ms_per_pass": round(narrow_total - wide_total, 6),
        "ideal_stage_specific_ms_per_pass": round(ideal_total, 6),
        "ideal_stage_specific_saving_ms_per_pass": round(ideal_saving, 6),
        "wide_emitted_per_pass": round(emitted_per_pass, 6),
        "ideal_saving_ms_per_output_token": round(
            ideal_saving / emitted_per_pass, 6
        ) if emitted_per_pass else None,
    }

    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
