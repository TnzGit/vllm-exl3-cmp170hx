#!/usr/bin/env python3
"""Read-only weight-footprint estimate for the Qwen3.8-Flash-Next EXL3 pack.

Groups safetensors header spans by family (text / vision / ngram / mtp) so the
first-boot gpu_memory_utilization can be chosen from measured bytes instead of
inherited numbers. Also reports the resident-vs-disk split implied by
VLLM_EXL3_NGRAM_TABLE=disk.
"""

from __future__ import annotations

import json
import struct
import sys
from pathlib import Path

DTYPE_BYTES = {
    "F64": 8, "F32": 4, "F16": 2, "BF16": 2, "I64": 8, "I32": 4,
    "I16": 2, "I8": 1, "U8": 1, "BOOL": 1,
}


def header(path: Path) -> dict:
    with path.open("rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
        return json.loads(fh.read(n))


def main() -> int:
    pack = Path(sys.argv[1]).expanduser().resolve()
    groups: dict[str, dict[str, float]] = {}
    per_file: dict[str, dict[str, float]] = {}

    for st in sorted(pack.glob("*.safetensors")):
        try:
            h = header(st)
        except Exception as exc:
            print(f"WARN: cannot read header {st.name}: {exc}", file=sys.stderr)
            continue
        acc: dict[str, float] = {}
        for name, meta in h.items():
            if name == "__metadata__":
                continue
            dtype = meta["dtype"]
            n = DTYPE_BYTES.get(dtype)
            if n is None:
                print(f"WARN: unknown dtype {dtype} for {name}", file=sys.stderr)
                continue
            span = meta["data_offsets"][1] - meta["data_offsets"][0]
            if "ngram_embedding" in name:
                g = "ngram"
            elif name.startswith("model.visual.") or name.startswith("visual."):
                g = "vision"
            elif name.startswith("mtp."):
                g = "mtp"
            else:
                g = "text"
            acc[g] = acc.get(g, 0.0) + span
            groups.setdefault(g, {"bytes": 0.0, "tensors": 0})
            groups[g]["bytes"] += span
            groups[g]["tensors"] += 1
        per_file[st.name] = acc

    gib = 2**30
    print("=== per-file byte spans (GiB) ===")
    for f, acc in per_file.items():
        print(f"  {f:42s} " + " ".join(f"{g}={v/gib:7.2f}" for g, v in sorted(acc.items())))

    print("=== group totals ===")
    for g, v in sorted(groups.items()):
        print(f"  {g:8s} tensors={v['tensors']:7d} bytes={v['bytes']:>14.0f} GiB={v['bytes']/gib:7.2f}")

    text = groups.get("text", {}).get("bytes", 0.0)
    mtp = groups.get("mtp", {}).get("bytes", 0.0)
    ngram = groups.get("ngram", {}).get("bytes", 0.0)
    vision = groups.get("vision", {}).get("bytes", 0.0)
    print("=== first-boot implications ===")
    print(f"  text-only resident (text+mtp, ngram on disk): {(text+mtp)/gib:7.2f} GiB")
    print(f"  + ngram resident (if VLLM_EXL3_NGRAM_TABLE=resident): {(text+mtp+ngram)/gib:7.2f} GiB")
    print(f"  dropped by --language-model-only (vision): {vision/gib:7.2f} GiB")
    print(f"  ngram table on disk (mmap): {ngram/gib:7.2f} GiB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
