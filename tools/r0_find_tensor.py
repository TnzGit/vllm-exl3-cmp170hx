#!/usr/bin/env python3
"""Locate safetensors tensors by shape, and report trellis K implied by words.

usage: python3 r0_find_tensor.py PACK [shape-substring ...]
"""

from __future__ import annotations

import json
import struct
import sys
from pathlib import Path


def header(path: Path) -> dict:
    with path.open("rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
        return json.loads(fh.read(n))


def main() -> int:
    pack = Path(sys.argv[1]).expanduser().resolve()
    want = sys.argv[2:]
    for st in sorted(pack.glob("*.safetensors")):
        try:
            h = header(st)
        except Exception as exc:
            print(f"WARN {st.name}: {exc}", file=sys.stderr)
            continue
        for name, meta in h.items():
            if name == "__metadata__":
                continue
            shape = meta["shape"]
            if len(shape) != 3:
                continue
            if not want:
                continue
            key = f"{shape[0]}, {shape[1]}, {shape[2]}"
            if any(w in key for w in want):
                print(f"{st.name}\t{name}\t{shape}\t{meta['dtype']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
