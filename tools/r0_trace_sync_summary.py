#!/usr/bin/env python3
"""Summarize synchronization/copy signatures from a torch-profiler trace.

Diagnostic only. This intentionally reports counts/durations and does not turn
profiler-on wall time into a production performance number.

usage:
  python3 r0_trace_sync_summary.py TRACE.json[.gz] [--passes N]
"""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path


NAMES = (
    "cudaStreamSynchronize",
    "cudaEventSynchronize",
    "cudaDeviceSynchronize",
    "cudaMemcpyAsync",
    "aten::item",
    "aten::_local_scalar_dense",
)


def load(path: Path) -> dict:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as fh:
        return json.load(fh)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("trace", type=Path)
    ap.add_argument("--passes", type=int, default=None)
    args = ap.parse_args()

    data = load(args.trace)
    events = data.get("traceEvents") or []
    out: dict[str, dict[str, float | int | None]] = {}
    for name in NAMES:
        selected = [e for e in events if e.get("name") == name]
        dur_us = sum(float(e.get("dur") or 0.0) for e in selected)
        out[name] = {
            "count": len(selected),
            "count_per_pass": (
                round(len(selected) / args.passes, 4) if args.passes else None
            ),
            "duration_ms": round(dur_us / 1000.0, 4),
            "duration_ms_per_pass": (
                round(dur_us / 1000.0 / args.passes, 4)
                if args.passes else None
            ),
        }

    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
