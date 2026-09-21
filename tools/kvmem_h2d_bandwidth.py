#!/usr/bin/env python3
"""Measure pinned host-to-device copy bandwidth on the local GPU."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import torch


def measure(size_mib: int, repeats: int) -> dict:
    n = size_mib * 1024 * 1024
    host = torch.empty(n, dtype=torch.uint8, pin_memory=True)
    dev = torch.empty(n, dtype=torch.uint8, device="cuda")
    host.fill_(17)

    for _ in range(3):
        dev.copy_(host, non_blocking=True)
    torch.cuda.synchronize()

    times_ms = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        dev.copy_(host, non_blocking=True)
        end.record()
        end.synchronize()
        times_ms.append(float(start.elapsed_time(end)))

    median_ms = statistics.median(times_ms)
    gib = n / (1024**3)
    gib_s = gib / (median_ms / 1000.0)
    return {
        "size_mib": size_mib,
        "repeats": repeats,
        "median_ms": median_ms,
        "min_ms": min(times_ms),
        "max_ms": max(times_ms),
        "effective_gib_s": gib_s,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sizes-mib", nargs="+", type=int, default=(64, 256, 512))
    ap.add_argument("--repeats", type=int, default=15)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    props = torch.cuda.get_device_properties(0)
    rows = [measure(size, args.repeats) for size in args.sizes_mib]
    large = max(rows, key=lambda x: x["size_mib"])
    bw = float(large["effective_gib_s"])
    estimates = {}
    for gib in (0.075, 0.15, 0.27, 0.285, 0.33):
        estimates[str(gib)] = {
            "gib": gib,
            "estimated_copy_ms_at_large_transfer_bandwidth": gib / bw * 1000.0,
        }

    result = {
        "schema": 1,
        "device": {
            "name": props.name,
            "capability": list(torch.cuda.get_device_capability(0)),
        },
        "measurements": rows,
        "reference_bandwidth_gib_s": bw,
        "stage_in_estimates": estimates,
        "note": (
            "CUDA-event pinned H2D copy time only. It excludes planner, "
            "allocation, block gather/scatter, synchronization and attention."
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
