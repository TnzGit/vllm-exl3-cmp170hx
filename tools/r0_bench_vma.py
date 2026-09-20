#!/usr/bin/env python3
"""Micro-benchmark _find_containing_vma against a realistic maps file.

The loader calls the madvise helper once per trellis and once per routed scale
tensor. Each call re-opens and line-parses /proc/self/maps. This measures that
cost directly with the checkpoint's shards mmapped, so the attribution does not
depend on a full 220 s model load.

usage: python3 r0_bench_vma.py PACK [--calls 3000]
"""

from __future__ import annotations

import argparse
import ctypes
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, "/home/base-node/.codex_tasks/qwen38-flashnext-r0/venv/lib/python3.12/site-packages")
from vllm_exl3.exl3 import _find_containing_vma  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("pack")
    ap.add_argument("--calls", type=int, default=3000)
    args = ap.parse_args()

    pack = Path(args.pack).expanduser().resolve()

    # Mirror the loader: mmap the safetensors shards before probing.
    maps = []
    libc = ctypes.CDLL("libc.so.6", use_errno=True)
    libc.mmap.restype = ctypes.c_void_p
    libc.mmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int,
                          ctypes.c_int, ctypes.c_int, ctypes.c_long]
    PROT_READ, MAP_PRIVATE = 0x1, 0x2
    for st in sorted(pack.glob("*.safetensors")):
        fd = os.open(st, os.O_RDONLY)
        size = os.fstat(fd).st_size
        addr = libc.mmap(None, size, PROT_READ, MAP_PRIVATE, fd, 0)
        maps.append((st.name, addr, size))
        os.close(fd)

    with open("/proc/self/maps") as fh:
        lines = fh.readlines()
    print(f"maps lines={len(lines)} shards_mmapped={len(maps)}")

    target_addr = maps[0][1]
    target = target_addr + (maps[0][2] // 2)

    t0 = time.perf_counter()
    hits = 0
    for _ in range(args.calls):
        if _find_containing_vma(target) is not None:
            hits += 1
    dt = time.perf_counter() - t0

    per_call_us = dt / args.calls * 1e6
    print(f"calls={args.calls} hits={hits} total={dt:.3f}s per_call={per_call_us:.2f}us")
    print(f"projected for 221,184 calls (73,728 trellis + 147,456 scales): "
          f"{per_call_us * 221184 / 1e6:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
