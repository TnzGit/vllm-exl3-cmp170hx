#!/usr/bin/env python3
"""Summarize lazy-safetensors tensor consumer attribution."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


GIB = 1024**3


def _merge_table(
    rows: list[dict],
    key: str,
) -> dict[str, dict[str, float | int]]:
    out: dict[str, dict[str, float | int]] = {}
    for shard in rows:
        for name, src in (shard.get(key) or {}).items():
            dst = out.setdefault(
                name,
                {
                    "count": 0,
                    "bytes": 0,
                    "get_tensor_wall_s": 0.0,
                    "get_tensor_cpu_s": 0.0,
                    "consumer_wall_s": 0.0,
                    "consumer_cpu_s": 0.0,
                },
            )
            for field in ("count", "bytes"):
                dst[field] = int(dst[field]) + int(src.get(field, 0))
            for field in (
                "get_tensor_wall_s",
                "get_tensor_cpu_s",
                "consumer_wall_s",
                "consumer_cpu_s",
            ):
                dst[field] = float(dst[field]) + float(src.get(field, 0.0))
    for dst in out.values():
        count = int(dst["count"])
        nbytes = int(dst["bytes"])
        dst["gib"] = nbytes / GIB
        dst["consumer_ms_per_tensor"] = (
            1000.0 * float(dst["consumer_wall_s"]) / count
            if count
            else None
        )
        dst["get_tensor_us_per_tensor"] = (
            1e6 * float(dst["get_tensor_wall_s"]) / count
            if count
            else None
        )
        dst["consumer_gib_s"] = (
            nbytes / GIB / float(dst["consumer_wall_s"])
            if nbytes and float(dst["consumer_wall_s"]) > 0
            else None
        )
    return out


def summarize(
    tensor_payload: dict,
    loader: dict,
) -> dict:
    rows = tensor_payload.get("rows") or []
    if not rows:
        raise ValueError("tensor attribution contains no shard rows")

    main_weights_s = float(loader["startup"]["main_weights_s"])
    exl3 = loader["exl3_trace"]

    total = {
        "shards": len(rows),
        "file_bytes": sum(int(r.get("file_bytes", 0)) for r in rows),
        "tensor_count": sum(int(r.get("tensor_count", 0)) for r in rows),
        "tensor_bytes": sum(int(r.get("tensor_bytes", 0)) for r in rows),
        "get_tensor_wall_s": sum(
            float(r.get("get_tensor_wall_s", 0.0)) for r in rows
        ),
        "get_tensor_cpu_s": sum(
            float(r.get("get_tensor_cpu_s", 0.0)) for r in rows
        ),
        "consumer_wall_s": sum(
            float(r.get("consumer_wall_s", 0.0)) for r in rows
        ),
        "consumer_cpu_s": sum(
            float(r.get("consumer_cpu_s", 0.0)) for r in rows
        ),
        "iterator_other_wall_s": sum(
            float(r.get("iterator_other_wall_s", 0.0)) for r in rows
        ),
        "shard_wall_s": sum(float(r.get("shard_wall_s", 0.0)) for r in rows),
    }
    total["file_gib"] = total["file_bytes"] / GIB
    total["tensor_gib"] = total["tensor_bytes"] / GIB
    total["consumer_fraction_of_main_weights"] = (
        total["consumer_wall_s"] / main_weights_s if main_weights_s else None
    )
    total["get_tensor_fraction_of_main_weights"] = (
        total["get_tensor_wall_s"] / main_weights_s if main_weights_s else None
    )
    total["shard_wall_fraction_of_main_weights"] = (
        total["shard_wall_s"] / main_weights_s if main_weights_s else None
    )
    total["main_weights_outside_shard_iterator_s"] = max(
        0.0, main_weights_s - total["shard_wall_s"]
    )

    by_suffix = _merge_table(rows, "by_suffix")
    by_scope = _merge_table(rows, "by_scope")
    by_size_bin = _merge_table(rows, "by_size_bin")
    by_scope_suffix = _merge_table(rows, "by_scope_suffix")

    top = []
    for row in rows:
        for item in row.get("top_consumers") or []:
            top.append({**item, "file": row.get("file")})
    top.sort(key=lambda x: float(x["consumer_wall_s"]), reverse=True)
    top = top[:40]

    trellis_consumer = float(
        (by_suffix.get("trellis") or {}).get("consumer_wall_s", 0.0)
    )
    exl3_meta_consumer = sum(
        float((by_suffix.get(suffix) or {}).get("consumer_wall_s", 0.0))
        for suffix in ("suh", "svh", "mcg", "mul1")
    )
    direct_copy = float(exl3.get("direct_trellis_copy_wall_s", 0.0))
    generic_copy = float(exl3.get("generic_copy_wall_s", 0.0))
    prep = float(exl3.get("direct_trellis_prep_wall_s", 0.0))

    measured_copy = direct_copy + generic_copy
    downstream_non_copy = max(0.0, total["consumer_wall_s"] - measured_copy)

    valid = bool(
        tensor_payload.get("mode") == "lazy_safetensors_tensor_consumer"
        and total["tensor_count"] > 0
        and total["consumer_wall_s"] > 0
        and total["shard_wall_s"] >= total["consumer_wall_s"]
        and main_weights_s > 0
    )

    return {
        "schema": 1,
        "tensor_consumer_attribution_valid": valid,
        "main_weights_s": main_weights_s,
        "totals": total,
        "by_suffix": by_suffix,
        "by_scope": by_scope,
        "by_size_bin": by_size_bin,
        "by_scope_suffix": by_scope_suffix,
        "top_consumers": top,
        "exl3_copy_reconciliation": {
            "trellis_consumer_wall_s": trellis_consumer,
            "direct_trellis_copy_wall_s": direct_copy,
            "direct_trellis_prep_wall_s": prep,
            "trellis_consumer_minus_direct_copy_s": (
                trellis_consumer - direct_copy
            ),
            "metadata_suffix_consumer_wall_s": exl3_meta_consumer,
            "generic_exl3_copy_wall_s": generic_copy,
            "metadata_consumer_minus_generic_copy_s": (
                exl3_meta_consumer - generic_copy
            ),
            "all_consumer_wall_s": total["consumer_wall_s"],
            "all_instrumented_exl3_copy_wall_s": measured_copy,
            "consumer_wall_outside_instrumented_exl3_copy_s": (
                downstream_non_copy
            ),
            "consumer_wall_outside_copy_fraction_of_main_weights": (
                downstream_non_copy / main_weights_s
            ),
        },
        "shards": rows,
        "interpretation_contract": {
            "get_tensor_wall": (
                "Time to create the safetensors tensor/view. With lazy mmap it "
                "does not imply all backing pages were materialized."
            ),
            "consumer_wall": (
                "Wall from yield to generator resume. It includes downstream "
                "mapping, AutoWeightsLoader dispatch, EXL3/native weight-loader "
                "work and any source-page faults triggered there."
            ),
            "consumer_cpu": (
                "Process CPU time accumulated while the yielded tensor is "
                "downstream. It may exceed wall when worker threads run."
            ),
            "copy_reconciliation": (
                "EXL3 copy timers are nested inside consumer wall and must be "
                "subtracted, not added."
            ),
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tensor-stats", type=Path, required=True)
    ap.add_argument("--loader-summary", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    out = summarize(
        json.loads(args.tensor_stats.read_text()),
        json.loads(args.loader_summary.read_text()),
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    return 0 if out["tensor_consumer_attribution_valid"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
