#!/usr/bin/env python3
"""Compare BASE / EARLY / EARLY+EMPTY k=3 Amdahl traces."""

from __future__ import annotations

import argparse
import json


BUCKETS = (
    "coop_moe_a",
    "coop_moe_b",
    "fill_zero",
    "index_scatter",
    "elementwise",
    "dtype_copy",
)


def metric(j: dict, bucket: str) -> tuple[float, float]:
    d = j["buckets"].get(bucket, {"ms": 0.0, "calls": 0})
    p = float(j["passes"])
    return float(d["ms"]) / p, float(d["calls"]) / p


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("base")
    ap.add_argument("early")
    ap.add_argument("empty")
    args = ap.parse_args()

    data = {
        "base": json.load(open(args.base)),
        "early": json.load(open(args.early)),
        "early_empty": json.load(open(args.empty)),
    }
    out: dict[str, object] = {}

    for bucket in BUCKETS:
        rows = {}
        base_ms, base_calls = metric(data["base"], bucket)
        for name, j in data.items():
            ms, calls = metric(j, bucket)
            rows[name] = {
                "ms_per_pass": round(ms, 6),
                "calls_per_pass": round(calls, 4),
                "delta_ms_vs_base": round(ms - base_ms, 6),
                "delta_calls_vs_base": round(calls - base_calls, 4),
                "delta_pct_vs_base": (
                    round((ms / base_ms - 1.0) * 100.0, 3)
                    if base_ms else None
                ),
            }
        out[bucket] = rows

    framework = ("fill_zero", "index_scatter", "elementwise")
    summary = {}
    base_fw = sum(metric(data["base"], b)[0] for b in framework)
    for name, j in data.items():
        fw = sum(metric(j, b)[0] for b in framework)
        moe = sum(metric(j, b)[0] for b in ("coop_moe_a", "coop_moe_b"))
        summary[name] = {
            "framework_prelude_ms_per_pass": round(fw, 6),
            "framework_delta_vs_base_ms_per_pass": round(fw - base_fw, 6),
            "coop_moe_ms_per_pass": round(moe, 6),
        }
    out["summary"] = summary

    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
