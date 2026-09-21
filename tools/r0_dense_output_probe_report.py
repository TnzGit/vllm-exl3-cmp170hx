#!/usr/bin/env python3
"""Compare the diagnostic FP16-output probe against the frozen 4K k=3 Amdahl.

The baseline constants are from docs/R0_MTP_K3_AMDAHL.md at
f2c6a719a1709ba40d08b97a4cc8d64b3c2256d8 (14 passes, 51 emitted).
This report compares per-pass GPU time, not profiler wall latency.

usage:
  python3 r0_dense_output_probe_report.py probe_amdahl.json
"""

from __future__ import annotations

import argparse
import json


BASE_MS_PER_PASS = {
    "dense_gemm": 5.243,
    "dense_gemv": 4.845,
    "dtype_copy": 2.701,
    "lm_head_candidate": 1.323,
}
BASE_CALLS_PER_PASS = {
    "dense_gemm": 145.0,
    "dense_gemv": 270.0,
    "dtype_copy": 833.0,
    "lm_head_candidate": 3.0,
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("probe_json")
    args = ap.parse_args()
    j = json.load(open(args.probe_json))
    passes = float(j["passes"])
    buckets = j["buckets"]

    out = {}
    for bucket, base_ms in BASE_MS_PER_PASS.items():
        d = buckets.get(bucket, {"ms": 0.0, "calls": 0})
        probe_ms = float(d["ms"]) / passes
        probe_calls = float(d["calls"]) / passes
        out[bucket] = {
            "base_ms_per_pass": base_ms,
            "probe_ms_per_pass": round(probe_ms, 6),
            "delta_ms_per_pass": round(probe_ms - base_ms, 6),
            "delta_pct": round((probe_ms / base_ms - 1.0) * 100.0, 3)
            if base_ms else None,
            "base_calls_per_pass": BASE_CALLS_PER_PASS[bucket],
            "probe_calls_per_pass": round(probe_calls, 4),
        }

    dense_boundary_base = (
        BASE_MS_PER_PASS["dense_gemm"]
        + BASE_MS_PER_PASS["dense_gemv"]
        + BASE_MS_PER_PASS["dtype_copy"]
    )
    dense_boundary_probe = sum(
        float(buckets.get(k, {"ms": 0.0})["ms"]) / passes
        for k in ("dense_gemm", "dense_gemv", "dtype_copy")
    )
    out["dense_plus_boundary"] = {
        "base_ms_per_pass": round(dense_boundary_base, 6),
        "probe_ms_per_pass": round(dense_boundary_probe, 6),
        "delta_ms_per_pass": round(dense_boundary_probe - dense_boundary_base, 6),
        "delta_pct": round(
            (dense_boundary_probe / dense_boundary_base - 1.0) * 100.0, 3
        ),
    }

    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
