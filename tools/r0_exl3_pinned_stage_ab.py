#!/usr/bin/env python3
"""Summarize one-boot shape-stratified direct-trellis pinned staging A/B."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


GIB = 1024**3


def _load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _find_tag(rows: list[dict], tag: str) -> dict | None:
    for row in rows:
        if row.get("tag") == tag:
            return row
    return None


def _delta_map(a: dict, b: dict) -> dict:
    out = {}
    for key in sorted(set(a) | set(b)):
        av = a.get(key, 0)
        bv = b.get(key, 0)
        if isinstance(av, (int, float)) and isinstance(bv, (int, float)):
            out[key] = bv - av
    return out


def _gib_s(nbytes: int, wall_s: float) -> float | None:
    if wall_s <= 0:
        return None
    return nbytes / GIB / wall_s


def _wall_per_gib(nbytes: int, wall_s: float) -> float | None:
    if nbytes <= 0:
        return None
    return wall_s / (nbytes / GIB)


def summarize(startup: dict, rows: list[dict], reference_h2d_gib_s: float) -> dict:
    first = _find_tag(rows, "FIRST_EXL3_COPY_BEFORE")
    final = _find_tag(rows, "ALL_WEIGHTS_LOADED_BEFORE_POSTLOAD")
    if first is None or final is None:
        raise ValueError("missing EXL3 loader trace boundaries")
    timing = _delta_map(
        first.get("loader_timing") or {},
        final.get("loader_timing") or {},
    )

    control_calls = int(timing.get("DIRECT_TRELLIS_CONTROL_CALLS", 0))
    control_bytes = int(timing.get("DIRECT_TRELLIS_CONTROL_BYTES", 0))
    control_wall = float(timing.get("DIRECT_TRELLIS_CONTROL_WALL_S", 0.0))
    pinned_calls = int(timing.get("DIRECT_TRELLIS_PINNED_CALLS", 0))
    pinned_bytes = int(timing.get("DIRECT_TRELLIS_PINNED_BYTES", 0))
    pinned_alloc_wall = float(
        timing.get("DIRECT_TRELLIS_PINNED_ALLOC_WALL_S", 0.0)
    )
    pinned_cpu_wall = float(
        timing.get("DIRECT_TRELLIS_PINNED_CPU_STAGE_WALL_S", 0.0)
    )
    pinned_h2d_wall = float(
        timing.get("DIRECT_TRELLIS_PINNED_H2D_WALL_S", 0.0)
    )
    pinned_total_wall = float(
        timing.get("DIRECT_TRELLIS_PINNED_TOTAL_WALL_S", 0.0)
    )
    pinned_buffer_max_bytes = int(
        final.get("loader_timing", {}).get(
            "DIRECT_TRELLIS_PINNED_BUFFER_MAX_BYTES", 0
        )
    )
    direct_total_bytes = int(timing.get("DIRECT_TRELLIS_COPY_BYTES", 0))
    direct_total_wall = float(timing.get("DIRECT_TRELLIS_COPY_WALL_S", 0.0))
    main_weights_s = float(startup["timings"]["main_weights_s"])

    control_wpg = _wall_per_gib(control_bytes, control_wall)
    pinned_wpg = _wall_per_gib(pinned_bytes, pinned_total_wall)
    control_gib_s = _gib_s(control_bytes, control_wall)
    pinned_stage_gib_s = _gib_s(pinned_bytes, pinned_cpu_wall)
    pinned_h2d_gib_s = _gib_s(pinned_bytes, pinned_h2d_wall)
    pinned_total_gib_s = _gib_s(pinned_bytes, pinned_total_wall)

    speedup = (
        control_wpg / pinned_wpg
        if control_wpg not in (None, 0) and pinned_wpg not in (None, 0)
        else None
    )
    pinned_share = (
        pinned_bytes / (control_bytes + pinned_bytes)
        if control_bytes + pinned_bytes
        else None
    )
    byte_balance_ok = pinned_share is not None and 0.45 <= pinned_share <= 0.55

    projected_all_control_direct = (
        direct_total_bytes / GIB * control_wpg
        if control_wpg is not None
        else None
    )
    projected_all_pinned_direct = (
        pinned_alloc_wall + direct_total_bytes / GIB * pinned_wpg
        if pinned_wpg is not None
        else None
    )
    mixed_direct_with_alloc = direct_total_wall + pinned_alloc_wall
    outside_direct_mixed = max(0.0, main_weights_s - mixed_direct_with_alloc)
    projected_all_control_main = (
        outside_direct_mixed + projected_all_control_direct
        if projected_all_control_direct is not None
        else None
    )
    projected_all_pinned_main = (
        outside_direct_mixed + projected_all_pinned_direct
        if projected_all_pinned_direct is not None
        else None
    )
    projected_main_saved = (
        projected_all_control_main - projected_all_pinned_main
        if projected_all_control_main is not None
        and projected_all_pinned_main is not None
        else None
    )

    valid = bool(
        final.get("pinned_stage_ab") is True
        and control_calls > 0
        and pinned_calls > 0
        and control_bytes > 0
        and pinned_bytes > 0
        and byte_balance_ok
        and abs(
            direct_total_bytes - (control_bytes + pinned_bytes)
        ) <= 4096
    )

    raw_floor = (
        direct_total_bytes / GIB / reference_h2d_gib_s
        if reference_h2d_gib_s > 0
        else None
    )

    return {
        "schema": 1,
        "pinned_stage_ab_valid": valid,
        "main_weights_s": main_weights_s,
        "reference_raw_h2d_gib_s": reference_h2d_gib_s,
        "direct_total": {
            "bytes": direct_total_bytes,
            "gib": direct_total_bytes / GIB,
            "wall_s": direct_total_wall,
            "raw_h2d_floor_s": raw_floor,
        },
        "control": {
            "calls": control_calls,
            "bytes": control_bytes,
            "gib": control_bytes / GIB,
            "wall_s": control_wall,
            "effective_gib_s": control_gib_s,
            "wall_s_per_gib": control_wpg,
        },
        "pinned": {
            "calls": pinned_calls,
            "bytes": pinned_bytes,
            "gib": pinned_bytes / GIB,
            "buffer_max_bytes": pinned_buffer_max_bytes,
            "buffer_max_mib": pinned_buffer_max_bytes / 2**20,
            "alloc_wall_s": pinned_alloc_wall,
            "cpu_stage_wall_s": pinned_cpu_wall,
            "cpu_stage_gib_s": pinned_stage_gib_s,
            "h2d_wall_s": pinned_h2d_wall,
            "h2d_gib_s": pinned_h2d_gib_s,
            "total_wall_s_ex_alloc": pinned_total_wall,
            "end_to_end_gib_s_ex_alloc": pinned_total_gib_s,
            "wall_s_per_gib_ex_alloc": pinned_wpg,
        },
        "balance": {
            "pinned_byte_share": pinned_share,
            "byte_balance_ok": byte_balance_ok,
            "direct_bytes_exact_partition": abs(
                direct_total_bytes - (control_bytes + pinned_bytes)
            ) <= 4096,
        },
        "comparison": {
            "pinned_vs_control_speedup": speedup,
            "pinned_wall_per_gib_ratio": (
                pinned_wpg / control_wpg
                if control_wpg not in (None, 0) and pinned_wpg is not None
                else None
            ),
            "projected_all_control_direct_wall_s": projected_all_control_direct,
            "projected_all_pinned_direct_wall_s": projected_all_pinned_direct,
            "projected_direct_saved_s": (
                projected_all_control_direct - projected_all_pinned_direct
                if projected_all_control_direct is not None
                and projected_all_pinned_direct is not None
                else None
            ),
            "outside_direct_mixed_s": outside_direct_mixed,
            "projected_all_control_main_weights_s": projected_all_control_main,
            "projected_all_pinned_main_weights_s": projected_all_pinned_main,
            "projected_main_weights_saved_s": projected_main_saved,
        },
        "interpretation_contract": {
            "single_boot_interleaved": True,
            "shape_stratified_selector": "arena shape-group idx parity",
            "pinned_total_excludes_allocation": True,
            "projection_is_not_qualification": True,
            "storage_and_copy_wall_overlap": True,
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--startup", type=Path, required=True)
    ap.add_argument("--trace", type=Path, required=True)
    ap.add_argument("--reference-h2d-gib-s", type=float, default=6.3494)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    out = summarize(
        json.loads(args.startup.read_text()),
        _load_jsonl(args.trace),
        args.reference_h2d_gib_s,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    return 0 if out["pinned_stage_ab_valid"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
