#!/usr/bin/env python3
"""Microbenchmark EXL3-sized CPU->GPU trellis copies on the local GPU.

This does not load a model. It compares:
  1. pageable CPU -> CUDA blocking copy (current direct-fill shape)
  2. pageable CPU -> CUDA nonblocking request
  3. persistent pinned staging ring -> CUDA async copy with event-owned reuse

The pinned-ring case explicitly keeps every source slot alive until its event
completes; it is a lifetime-safe prototype for a possible loader pipeline.

Example:
  python tools/r0_bench_trellis_h2d.py --bytes 655360 --calls 4096 --slots 4 8 16 32
"""

from __future__ import annotations

import argparse
import statistics
import time

import torch


def gib_per_s(total_bytes: int, seconds: float) -> float:
    return total_bytes / (1024**3) / max(seconds, 1e-12)


def bench_blocking(src: torch.Tensor, dst: torch.Tensor, calls: int) -> float:
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(calls):
        dst.copy_(src, non_blocking=False)
    torch.cuda.synchronize()
    return time.perf_counter() - t0


def bench_pageable_nonblocking(
    src: torch.Tensor, dst: torch.Tensor, calls: int
) -> float:
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(calls):
        dst.copy_(src, non_blocking=True)
    torch.cuda.synchronize()
    return time.perf_counter() - t0


def bench_pinned_ring(
    src: torch.Tensor,
    dst: torch.Tensor,
    calls: int,
    slots: int,
) -> float:
    ring = [
        torch.empty_like(src, device="cpu", pin_memory=True)
        for _ in range(slots)
    ]
    events = [torch.cuda.Event(blocking=False) for _ in range(slots)]
    armed = [False] * slots
    stream = torch.cuda.current_stream()

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for i in range(calls):
        slot = i % slots
        if armed[slot]:
            events[slot].synchronize()
        # CPU copy completes before the pinned slot is submitted to CUDA.
        ring[slot].copy_(src, non_blocking=False)
        dst.copy_(ring[slot], non_blocking=True)
        events[slot].record(stream)
        armed[slot] = True

    for i, event in enumerate(events):
        if armed[i]:
            event.synchronize()
    torch.cuda.synchronize()
    return time.perf_counter() - t0


def summarize(label: str, samples: list[float], total_bytes: int, calls: int) -> None:
    med = statistics.median(samples)
    per_us = med / calls * 1e6
    print(
        f"{label:28s} median={med:8.4f}s "
        f"per_call={per_us:9.2f}us "
        f"throughput={gib_per_s(total_bytes, med):7.3f} GiB/s "
        f"runs={[round(x, 4) for x in samples]}"
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bytes", type=int, default=640 * 1024)
    ap.add_argument("--calls", type=int, default=4096)
    ap.add_argument("--repeat", type=int, default=3)
    ap.add_argument("--slots", type=int, nargs="+", default=[4, 8, 16, 32])
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA unavailable")
    if args.bytes <= 0 or args.calls <= 0 or args.repeat <= 0:
        raise SystemExit("bytes/calls/repeat must be >0")

    # Trellis payload is int16. Round down to a valid element count.
    elems = args.bytes // 2
    nbytes = elems * 2
    src = torch.arange(elems, dtype=torch.int16, device="cpu")
    dst = torch.empty(elems, dtype=torch.int16, device="cuda")
    total = nbytes * args.calls

    print(
        f"device={torch.cuda.get_device_name()} bytes_per_call={nbytes} "
        f"calls={args.calls} total={total / 1024**3:.3f} GiB"
    )

    # Warm CUDA context and pageable transfer machinery.
    dst.copy_(src, non_blocking=False)
    torch.cuda.synchronize()

    blocking = [
        bench_blocking(src, dst, args.calls) for _ in range(args.repeat)
    ]
    summarize("pageable blocking", blocking, total, args.calls)

    pageable_nb = [
        bench_pageable_nonblocking(src, dst, args.calls)
        for _ in range(args.repeat)
    ]
    summarize("pageable nonblocking", pageable_nb, total, args.calls)

    for slots in args.slots:
        if slots <= 0:
            continue
        samples = [
            bench_pinned_ring(src, dst, args.calls, slots)
            for _ in range(args.repeat)
        ]
        summarize(f"pinned ring slots={slots}", samples, total, args.calls)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
