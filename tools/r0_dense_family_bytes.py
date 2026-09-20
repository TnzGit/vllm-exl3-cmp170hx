#!/usr/bin/env python3
"""Aggregate dense EXL3 weight bytes by module family from safetensors headers.

GEMV decode is weight-bound, so weight bytes per family is the right proxy for
which dense linear families dominate decode cost. Reads headers only.

usage: python3 r0_dense_family_bytes.py PACK
"""

from __future__ import annotations

import collections
import glob
import json
import os
import re
import struct
import sys


def hdr(path: str) -> dict:
    with open(path, "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
        return json.loads(fh.read(n))


def main() -> int:
    pack = os.path.expanduser(sys.argv[1])
    agg: dict[str, list] = collections.defaultdict(lambda: [0, 0.0, 0])
    for st in sorted(glob.glob(os.path.join(pack, "*.safetensors"))):
        for name, meta in hdr(st).items():
            if name == "__metadata__" or not name.endswith(".trellis"):
                continue
            base = name[: -len(".trellis")]
            if ".experts." in base or "ngram" in base:
                continue
            a, b, w = meta["shape"]
            inn, out, k = a * 16, b, w // 16
            fam = re.sub(r"\.layers\.\d+\.", "..layers..", base)
            fam = re.sub(r"\.blocks\.\d+\.", "..blocks..", fam)
            agg[fam][0] += 1
            agg[fam][1] += inn * out * k / 8.0
            agg[fam][2] = k

    total = sum(v[1] for v in agg.values())
    print(f"total dense EXL3 weight bytes = {total / 2**20:.1f} MiB")
    print(f"{'family':52s} {'n':>4s} {'MiB':>9s} {'K':>2s} {'pct':>6s}")
    for fam, (n, by, k) in sorted(agg.items(), key=lambda x: -x[1][1]):
        print(f"{fam:52s} {n:4d} {by / 2**20:9.2f} {k:2d} {100 * by / total:6.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
