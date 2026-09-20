#!/usr/bin/env python3
"""Attribute post-COOP eager CUDA kernels back to CPU/Python launch context.

This is diagnostic-only tooling for vLLM --enforce-eager + torch.profiler
traces. It focuses on the residual families that were unattributed under CUDA
graph replay:

  - cuBLAS gemv2T / gemvx
  - cutlass_80_wmma
  - direct/bfloat16 copy kernels
  - memcpy32_post
  - FillFunctor / compare_scalar / index_elementwise / scatter-gather
  - persistent_topk / bitonic sort

The trace remains the source of truth. This script never converts eager timing
into a production performance claim.
"""

from __future__ import annotations

import argparse
from bisect import bisect_right
from collections import defaultdict
import gzip
import json
from pathlib import Path
from typing import Any


TARGETS = (
    "gemv2t_kernel",
    "gemvx::kernel",
    "cutlass_80_wmma",
    "direct_copy_kernel",
    "bfloat16_copy_kernel",
    "memcpy32_post",
    "fillfunctor",
    "compare_scalar_kernel",
    "index_elementwise_kernel",
    "_scatter_gather_elementwise",
    "persistent_topk",
    "bitonicsort",
)


def _load(path: Path) -> dict[str, Any]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as fh:
        return json.load(fh)


def _arg_int(args: dict[str, Any], *names: str) -> int | None:
    lowered = {str(k).lower().replace("_", " "): v for k, v in args.items()}
    for name in names:
        value = lowered.get(name.lower().replace("_", " "))
        if isinstance(value, bool):
            continue
        if isinstance(value, (int, float)):
            return int(value)
        if isinstance(value, str):
            try:
                return int(value, 0)
            except ValueError:
                pass
    return None


def _is_target_kernel(name: str) -> bool:
    n = name.lower()
    return any(token in n for token in TARGETS)


def _event_name(ev: dict[str, Any]) -> str:
    return str(ev.get("name", "<unnamed>"))


def _event_interval(ev: dict[str, Any]) -> tuple[float, float]:
    start = float(ev.get("ts", 0.0))
    dur = float(ev.get("dur", 0.0))
    return start, start + max(dur, 0.0)


def _pythonish(cat: str) -> bool:
    c = cat.lower()
    return "python" in c or "user_annotation" in c


def _stackish_name(name: str) -> bool:
    n = name.lower()
    return (
        "forward" in n
        or "linear" in n
        or "qwen" in n
        or "attention" in n
        or "hyper" in n
        or "sampling" in n
        or "topk" in n
        or "norm" in n
        or "index" in n
    )


def _nearest_python_parent(
    cpu_ev: dict[str, Any],
    python_events_by_tid: dict[tuple[int, int], list[dict[str, Any]]],
) -> str:
    key = (int(cpu_ev.get("pid", -1)), int(cpu_ev.get("tid", -1)))
    events = python_events_by_tid.get(key) or []
    if not events:
        return "<no-python-parent>"

    c0, c1 = _event_interval(cpu_ev)
    starts = [float(ev.get("ts", 0.0)) for ev in events]
    pos = bisect_right(starts, c0)
    best: dict[str, Any] | None = None
    best_dur = float("inf")

    # Nearby containing events are enough because events are sorted by start.
    for idx in range(max(0, pos - 64), pos):
        ev = events[idx]
        p0, p1 = _event_interval(ev)
        if p0 <= c0 and p1 >= c1:
            dur = p1 - p0
            name = _event_name(ev)
            # Prefer the tightest useful model/module-looking parent.
            bonus = 0.25 if _stackish_name(name) else 1.0
            score = dur * bonus
            if score < best_dur:
                best = ev
                best_dur = score
    return _event_name(best) if best is not None else "<no-python-parent>"


def _print_rows(
    title: str,
    rows: dict[str, list[float | int]],
    total_us: float,
    limit: int,
) -> None:
    print(f"\n## {title}")
    for name, values in sorted(
        rows.items(), key=lambda item: float(item[1][1]), reverse=True
    )[:limit]:
        count = int(values[0])
        dur_us = float(values[1])
        pct = 0.0 if total_us <= 0 else dur_us / total_us * 100.0
        print(f"{dur_us/1000:10.3f} ms  {pct:6.2f}%  {count:8d}  {name}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("trace", type=Path)
    ap.add_argument("--top", type=int, default=80)
    ap.add_argument("--steps", type=float, default=None)
    ap.add_argument("--layers", type=int, default=48)
    args = ap.parse_args()

    obj = _load(args.trace)
    raw_events = obj.get("traceEvents") or []
    events = [
        ev for ev in raw_events
        if isinstance(ev, dict)
        and ev.get("ph") == "X"
        and isinstance(ev.get("dur"), (int, float))
        and float(ev.get("dur", 0.0)) > 0
    ]

    cpu_by_external: dict[int, dict[str, Any]] = {}
    runtime_external_by_corr: dict[int, int] = {}
    python_events_by_tid: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    kernels: list[dict[str, Any]] = []

    for ev in events:
        cat = str(ev.get("cat", "")).lower()
        ev_args = ev.get("args") if isinstance(ev.get("args"), dict) else {}

        if "cpu_op" in cat:
            external_id = _arg_int(ev_args, "external id", "external_id")
            if external_id is not None:
                cpu_by_external.setdefault(external_id, ev)

        if "runtime" in cat:
            corr = _arg_int(ev_args, "correlation", "correlation id")
            external_id = _arg_int(ev_args, "external id", "external_id")
            if corr is not None and external_id is not None:
                runtime_external_by_corr.setdefault(corr, external_id)

        if _pythonish(cat):
            key = (int(ev.get("pid", -1)), int(ev.get("tid", -1)))
            python_events_by_tid[key].append(ev)

        if "kernel" in cat:
            kernels.append(ev)

    for evs in python_events_by_tid.values():
        evs.sort(key=lambda ev: float(ev.get("ts", 0.0)))

    total_kernel_us = sum(float(ev["dur"]) for ev in kernels)
    coop_a_count = sum(
        1 for ev in kernels
        if "exl3_moe_coop_a_kernel" in _event_name(ev).lower()
    )
    inferred_steps: float | None = args.steps
    if inferred_steps is None and args.layers > 0 and coop_a_count > 0:
        inferred_steps = coop_a_count / args.layers

    target_kernel_us = 0.0
    mapped_us = 0.0

    by_kernel: dict[str, list[float | int]] = defaultdict(lambda: [0, 0.0])
    by_cpu: dict[str, list[float | int]] = defaultdict(lambda: [0, 0.0])
    by_parent: dict[str, list[float | int]] = defaultdict(lambda: [0, 0.0])
    by_pair: dict[str, list[float | int]] = defaultdict(lambda: [0, 0.0])
    by_triplet: dict[str, list[float | int]] = defaultdict(lambda: [0, 0.0])

    for ev in kernels:
        kernel = _event_name(ev)
        if not _is_target_kernel(kernel):
            continue
        dur_us = float(ev["dur"])
        target_kernel_us += dur_us
        by_kernel[kernel][0] += 1
        by_kernel[kernel][1] += dur_us

        ev_args = ev.get("args") if isinstance(ev.get("args"), dict) else {}
        corr = _arg_int(ev_args, "correlation", "correlation id")
        if corr is None:
            continue
        external_id = runtime_external_by_corr.get(corr)
        if external_id is None:
            continue
        cpu_ev = cpu_by_external.get(external_id)
        if cpu_ev is None:
            continue

        mapped_us += dur_us
        cpu_name = _event_name(cpu_ev)
        parent = _nearest_python_parent(cpu_ev, python_events_by_tid)

        by_cpu[cpu_name][0] += 1
        by_cpu[cpu_name][1] += dur_us
        by_parent[parent][0] += 1
        by_parent[parent][1] += dur_us

        pair = f"{cpu_name}  ->  {kernel}"
        by_pair[pair][0] += 1
        by_pair[pair][1] += dur_us

        triplet = f"{parent}  ::  {cpu_name}  ->  {kernel}"
        by_triplet[triplet][0] += 1
        by_triplet[triplet][1] += dur_us

    print(f"trace={args.trace}")
    print(f"total_gpu_kernel_ms={total_kernel_us/1000:.3f}")
    print(f"target_gpu_kernel_ms={target_kernel_us/1000:.3f}")
    print(
        "target_share_of_all_gpu="
        f"{(100.0 * target_kernel_us / total_kernel_us if total_kernel_us else 0.0):.2f}%"
    )
    print(
        "target_correlation_coverage="
        f"{(100.0 * mapped_us / target_kernel_us if target_kernel_us else 0.0):.2f}%"
    )
    print(f"coop_a_kernel_count={coop_a_count}")
    if inferred_steps is not None and inferred_steps > 0:
        print(f"decode_steps={inferred_steps:.3f}")
        print(f"target_ms_per_step={target_kernel_us / inferred_steps / 1000:.4f}")

    _print_rows("target raw kernels", by_kernel, target_kernel_us, args.top)
    _print_rows(
        "target GPU time by correlated CPU op",
        by_cpu,
        target_kernel_us,
        args.top,
    )
    _print_rows(
        "target GPU time by nearest Python/function parent",
        by_parent,
        target_kernel_us,
        args.top,
    )
    _print_rows(
        "target CPU-op -> kernel pairs",
        by_pair,
        target_kernel_us,
        args.top,
    )
    _print_rows(
        "target Python-parent :: CPU-op -> kernel triplets",
        by_triplet,
        target_kernel_us,
        args.top,
    )

    if target_kernel_us <= 0:
        raise SystemExit("No target residual kernels found in trace")
    if mapped_us <= 0:
        raise SystemExit(
            "Target kernels were found but none correlated back to CPU ops; "
            "verify the server truly ran with --enforce-eager."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
