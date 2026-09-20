#!/usr/bin/env python3
"""Summarize vLLM/PyTorch Chrome trace files for decode Amdahl work.

Accepts .json or .json.gz files. Prints category totals, top CUDA kernels and
simple component buckets. Bucket labels are heuristic; raw top-event output is
the primary evidence.
"""

from __future__ import annotations

import argparse
import gzip
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def _load(path: Path) -> dict[str, Any]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as fh:
        return json.load(fh)


def _bucket(name: str) -> str:
    n = name.lower()
    if "ngram" in n or "ple" in n:
        return "PLE/NGRAM"
    if "qsa" in n or "indexer" in n:
        return "QSA/INDEXER"
    if "gdn" in n or "gated_delta" in n or "short_conv" in n:
        return "GDN/RECURRENT"
    if "exl3_moe" in n or ("exl3" in n and "moe" in n):
        return "EXL3_MOE_COOP" if "coop" in n else "EXL3_MOE"
    if "_hc_" in n or "qwen4_exp_hc" in n or "hyperconnection" in n:
        return "HYPER_CONNECTION"
    if "exl3" in n:
        return "EXL3_OTHER"
    if (
        "gemv2t_kernel" in n
        or "gemvx::kernel" in n
        or "cutlass_80_wmma" in n
    ):
        # These are real BF16 GEMM/GEMV costs, but the raw kernel name does not
        # prove whether the caller is HC, attention projection, LM head, etc.
        return "BF16_GEMM_UNATTRIBUTED"
    if (
        "direct_copy_kernel" in n
        or "bfloat16_copy_kernel" in n
        or "memcpy32_post" in n
        or "fillfunctor" in n
        or "compare_scalar_kernel" in n
        or "index_elementwise_kernel" in n
        or "_scatter_gather_elementwise" in n
    ):
        return "FRAMEWORK_ELEMENTWISE"
    if "persistent_topk" in n or "bitonicsort" in n:
        # Counts can resemble QSA layers, but do not silently attribute a
        # generic top-k/sort kernel without correlation evidence.
        return "TOPK_SORT_UNATTRIBUTED"
    if "gemm" in n or "mma" in n or "matmul" in n:
        return "GEMM_OTHER"
    return "OTHER"


def _print_top(title: str, rows: dict[str, tuple[int, float]], limit: int) -> None:
    print(f"\n## {title}")
    total = sum(v[1] for v in rows.values())
    print(f"total_ms={total / 1000.0:.3f}")
    for name, (count, dur_us) in sorted(
        rows.items(), key=lambda kv: kv[1][1], reverse=True
    )[:limit]:
        pct = 0.0 if total <= 0 else dur_us / total * 100.0
        print(
            f"{dur_us / 1000.0:10.3f} ms  {pct:6.2f}%  "
            f"{count:8d}  {name}"
        )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("trace", type=Path)
    ap.add_argument("--top", type=int, default=40)
    ap.add_argument("--steps", type=float, default=None, help="optional decode-step count for per-step totals")
    args = ap.parse_args()

    obj = _load(args.trace)
    events = obj.get("traceEvents") or []

    categories: dict[str, list[float | int]] = defaultdict(lambda: [0, 0.0])
    kernels: dict[str, list[float | int]] = defaultdict(lambda: [0, 0.0])
    cpu_ops: dict[str, list[float | int]] = defaultdict(lambda: [0, 0.0])
    buckets: dict[str, list[float | int]] = defaultdict(lambda: [0, 0.0])

    for ev in events:
        if ev.get("ph") != "X":
            continue
        dur = ev.get("dur")
        if not isinstance(dur, (int, float)) or dur <= 0:
            continue
        name = str(ev.get("name", "<unnamed>"))
        cat = str(ev.get("cat", "")).lower()
        categories[cat][0] += 1
        categories[cat][1] += float(dur)

        if "kernel" in cat:
            kernels[name][0] += 1
            kernels[name][1] += float(dur)
            bucket = _bucket(name)
            buckets[bucket][0] += 1
            buckets[bucket][1] += float(dur)
        elif cat == "cpu_op" or "cpu_op" in cat:
            cpu_ops[name][0] += 1
            cpu_ops[name][1] += float(dur)

    _print_top(
        "event categories",
        {k: (int(v[0]), float(v[1])) for k, v in categories.items()},
        args.top,
    )
    _print_top(
        "CUDA kernels",
        {k: (int(v[0]), float(v[1])) for k, v in kernels.items()},
        args.top,
    )
    _print_top(
        "heuristic GPU component buckets",
        {k: (int(v[0]), float(v[1])) for k, v in buckets.items()},
        args.top,
    )
    _print_top(
        "CPU operators",
        {k: (int(v[0]), float(v[1])) for k, v in cpu_ops.items()},
        args.top,
    )

    if args.steps is not None:
        if args.steps <= 0:
            raise SystemExit("--steps must be > 0")
        kernel_us = sum(float(v[1]) for v in kernels.values())
        print(
            f"\nGPU kernel total per decode step: "
            f"{kernel_us / args.steps / 1000.0:.4f} ms "
            f"({args.steps:g} steps)"
        )

    if not kernels:
        raise SystemExit(
            "No cat=kernel events found. Verify the worker trace contains CUDA "
            "activities and that this is the EngineCore/worker trace."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
