#!/usr/bin/env python3
"""Summarize EXL3 startup loader attribution evidence."""

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


def summarize(
    startup: dict,
    trace_rows: list[dict],
    proc_watch: dict,
    reference_h2d_gib_s: float,
) -> dict:
    first = _find_tag(trace_rows, "FIRST_EXL3_COPY_BEFORE")
    final = _find_tag(trace_rows, "ALL_WEIGHTS_LOADED_BEFORE_POSTLOAD")
    if first is None or final is None:
        raise ValueError("missing EXL3 loader trace boundaries")

    timing = _delta_map(
        first.get("loader_timing") or {},
        final.get("loader_timing") or {},
    )
    direct = _delta_map(
        first.get("direct_fill") or {},
        final.get("direct_fill") or {},
    )
    trace_io = _delta_map(
        (first.get("proc_io") or {}),
        (final.get("proc_io") or {}),
    )

    main_weights_s = startup["timings"]["main_weights_s"]
    draft_weights_s = startup["timings"].get("draft_weights_s")
    checkpoint_gib = startup["checkpoint"]["checkpoint_gib"]

    generic_bytes = int(timing.get("GENERIC_COPY_BYTES", 0))
    direct_bytes = int(timing.get("DIRECT_TRELLIS_COPY_BYTES", 0))
    instrumented_copy_bytes = generic_bytes + direct_bytes
    generic_wall = float(timing.get("GENERIC_COPY_WALL_S", 0.0))
    direct_wall = float(timing.get("DIRECT_TRELLIS_COPY_WALL_S", 0.0))
    prep_wall = float(timing.get("DIRECT_TRELLIS_PREP_WALL_S", 0.0))
    instrumented_copy_wall = generic_wall + direct_wall

    raw_floor_s = (
        instrumented_copy_bytes / GIB / reference_h2d_gib_s
        if reference_h2d_gib_s > 0
        else None
    )
    copy_effective_gib_s = (
        instrumented_copy_bytes / GIB / instrumented_copy_wall
        if instrumented_copy_wall > 0
        else None
    )

    kernel_window = (
        proc_watch.get("deltas", {}).get("model_start_to_main_weights_done")
        or proc_watch.get("deltas", {}).get("enginecore_to_main_weights_done")
    )
    kernel_io = (kernel_window or {}).get("io") or {}
    kernel_stat = (kernel_window or {}).get("stat") or {}
    kernel_read_bytes = int(kernel_io.get("read_bytes", 0))

    fallback_calls = int(
        final.get("direct_fill", {}).get("DIRECT_FILL_FALLBACK_CALLS", 0)
    )
    direct_calls = int(
        final.get("direct_fill", {}).get("DIRECT_FILL_CALLS", 0)
    )
    trace_same_pid = first.get("pid") == final.get("pid")
    boundaries_ok = bool(
        trace_same_pid
        and final["monotonic_s"] >= first["monotonic_s"]
        and direct_calls > 0
        and instrumented_copy_bytes > 0
    )

    return {
        "schema": 1,
        "loader_attribution_valid": boundaries_ok,
        "startup": {
            "main_weights_s": main_weights_s,
            "draft_weights_s": draft_weights_s,
            "total_weights_s": startup["timings"].get("total_weights_s"),
            "model_construct_postload_s": startup["timings"].get(
                "model_construct_postload_s"
            ),
            "checkpoint_gib": checkpoint_gib,
        },
        "exl3_trace": {
            "pid": first.get("pid"),
            "first_copy_monotonic_s": first["monotonic_s"],
            "all_weights_loaded_monotonic_s": final["monotonic_s"],
            "elapsed_first_copy_to_all_weights_loaded_s": (
                final["monotonic_s"] - first["monotonic_s"]
            ),
            "direct_fill_calls": direct_calls,
            "direct_fill_bytes": int(
                final.get("direct_fill", {}).get("DIRECT_FILL_BYTES", 0)
            ),
            "direct_fill_fallback_calls": fallback_calls,
            "direct_fill_fallback_bytes": int(
                final.get("direct_fill", {}).get(
                    "DIRECT_FILL_FALLBACK_BYTES", 0
                )
            ),
            "generic_copy_calls": int(timing.get("GENERIC_COPY_CALLS", 0)),
            "generic_copy_bytes": generic_bytes,
            "generic_copy_wall_s": generic_wall,
            "direct_trellis_prep_calls": int(
                timing.get("DIRECT_TRELLIS_PREP_CALLS", 0)
            ),
            "direct_trellis_prep_bytes": int(
                timing.get("DIRECT_TRELLIS_PREP_BYTES", 0)
            ),
            "direct_trellis_prep_wall_s": prep_wall,
            "direct_trellis_copy_calls": int(
                timing.get("DIRECT_TRELLIS_COPY_CALLS", 0)
            ),
            "direct_trellis_copy_bytes": direct_bytes,
            "direct_trellis_copy_wall_s": direct_wall,
            "instrumented_copy_bytes": instrumented_copy_bytes,
            "instrumented_copy_gib": instrumented_copy_bytes / GIB,
            "instrumented_copy_wall_s": instrumented_copy_wall,
            "instrumented_copy_effective_gib_s": copy_effective_gib_s,
            "reference_raw_h2d_gib_s": reference_h2d_gib_s,
            "raw_h2d_floor_for_instrumented_bytes_s": raw_floor_s,
            "copy_wall_over_raw_h2d_floor": (
                instrumented_copy_wall / raw_floor_s
                if raw_floor_s not in (None, 0)
                else None
            ),
            "copy_wall_fraction_of_main_weights": (
                instrumented_copy_wall / main_weights_s
                if main_weights_s
                else None
            ),
            "trace_proc_io_delta": trace_io,
            "trace_ru_minflt_delta": (
                int(final.get("ru_minflt", 0)) - int(first.get("ru_minflt", 0))
            ),
            "trace_ru_majflt_delta": (
                int(final.get("ru_majflt", 0)) - int(first.get("ru_majflt", 0))
            ),
            "trace_ru_inblock_delta": (
                int(final.get("ru_inblock", 0)) - int(first.get("ru_inblock", 0))
            ),
        },
        "kernel_model_load_window": {
            "source_window": (
                "model_start_to_main_weights_done"
                if proc_watch.get("deltas", {}).get(
                    "model_start_to_main_weights_done"
                )
                else "enginecore_to_main_weights_done"
            ),
            "elapsed_s": (kernel_window or {}).get("elapsed_s"),
            "read_bytes": kernel_read_bytes,
            "read_gib": kernel_read_bytes / GIB,
            "rchar_bytes": int(kernel_io.get("rchar", 0)),
            "rchar_gib": int(kernel_io.get("rchar", 0)) / GIB,
            "syscr": int(kernel_io.get("syscr", 0)),
            "major_faults": int(kernel_stat.get("majflt", 0)),
            "minor_faults": int(kernel_stat.get("minflt", 0)),
            "user_cpu_s": float(kernel_stat.get("utime_s", 0.0)),
            "system_cpu_s": float(kernel_stat.get("stime_s", 0.0)),
            "kernel_read_gib_per_main_weight_s": (
                kernel_read_bytes / GIB / main_weights_s
                if main_weights_s
                else None
            ),
            "kernel_read_vs_checkpoint_ratio": (
                kernel_read_bytes / GIB / checkpoint_gib
                if checkpoint_gib
                else None
            ),
        },
        "interpretation_contract": {
            "kernel_read_bytes": (
                "Linux /proc/<EngineCore>/io read_bytes: storage-layer bytes "
                "charged to the process, not a pure safetensors byte counter."
            ),
            "copy_wall": (
                "Blocking EXL3 destination copy call wall time. With mmap/pageable "
                "sources it may include source-page faults plus H2D; compare against "
                "the pinned raw-H2D floor, do not add storage time to copy time."
            ),
            "outside_instrumented_copy_calls_s": (
                max(0.0, main_weights_s - instrumented_copy_wall)
                if main_weights_s is not None
                else None
            ),
            "direct_fill_no_fallback": fallback_calls == 0,
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--startup", type=Path, required=True)
    ap.add_argument("--trace", type=Path, required=True)
    ap.add_argument("--proc-watch", type=Path, required=True)
    ap.add_argument("--reference-h2d-gib-s", type=float, default=6.3494)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    out = summarize(
        json.loads(args.startup.read_text()),
        _load_jsonl(args.trace),
        json.loads(args.proc_watch.read_text()),
        args.reference_h2d_gib_s,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    return 0 if out["loader_attribution_valid"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
