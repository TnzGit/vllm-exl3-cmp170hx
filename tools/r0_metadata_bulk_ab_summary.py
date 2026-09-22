#!/usr/bin/env python3
"""Summarize interleaved routed-expert metadata bulk-commit A/B."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _load_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text().splitlines()
        if line.strip()
    ]


def _final_trace(rows: list[dict]) -> dict:
    for row in rows:
        if row.get("tag") == "ALL_WEIGHTS_LOADED_BEFORE_POSTLOAD":
            return row
    raise ValueError("missing ALL_WEIGHTS_LOADED_BEFORE_POSTLOAD trace record")


def _suffix_rows(stats: dict) -> dict[str, dict]:
    control = stats.get("control_by_suffix") or {}
    deferred = stats.get("deferred_by_suffix") or {}
    commit = stats.get("commit_by_suffix") or {}
    out = {}
    for suffix in sorted(set(control) | set(deferred) | set(commit)):
        c = control.get(suffix) or {}
        d = deferred.get(suffix) or {}
        b = commit.get(suffix) or {}
        c_wall = float(c.get("wall_s", 0.0))
        d_stage = float(d.get("wall_s", 0.0))
        b_wall = float(b.get("wall_s", 0.0))
        d_total = d_stage + b_wall
        c_calls = int(c.get("calls", 0))
        d_calls = int(d.get("calls", 0))
        c_bytes = int(c.get("bytes", 0))
        d_bytes = int(d.get("bytes", 0))
        b_bytes = int(b.get("bytes", 0))
        out[suffix] = {
            "control_calls": c_calls,
            "control_bytes": c_bytes,
            "control_wall_s": c_wall,
            "deferred_calls": d_calls,
            "deferred_bytes": d_bytes,
            "deferred_stage_wall_s": d_stage,
            "bulk_commit_calls": int(b.get("calls", 0)),
            "bulk_commit_bytes": b_bytes,
            "bulk_commit_wall_s": b_wall,
            "deferred_total_wall_s": d_total,
            "calls_balanced": c_calls == d_calls and c_calls > 0,
            "bytes_balanced": c_bytes == d_bytes and c_bytes > 0,
            "commit_bytes_exact": b_bytes == d_bytes and d_bytes > 0,
            "control_ms_per_tensor": (
                1000.0 * c_wall / c_calls if c_calls else None
            ),
            "deferred_ms_per_tensor_including_commit": (
                1000.0 * d_total / d_calls if d_calls else None
            ),
            "speedup": (
                c_wall / d_total
                if c_wall > 0 and d_total > 0 and c_calls == d_calls
                else None
            ),
            "gpu_call_reduction": (
                d_calls / int(b.get("calls", 0))
                if int(b.get("calls", 0)) > 0
                else None
            ),
        }
    return out


def summarize(trace_rows: list[dict], tensor_summary: dict, loader: dict) -> dict:
    final = _final_trace(trace_rows)
    stats = final.get("metadata_bulk_ab") or {}
    suffix = _suffix_rows(stats)

    control_calls = int(stats.get("control_calls", 0))
    deferred_calls = int(stats.get("deferred_calls", 0))
    control_bytes = int(stats.get("control_bytes", 0))
    deferred_bytes = int(stats.get("deferred_bytes", 0))
    commit_bytes = int(stats.get("commit_bytes", 0))
    control_wall = float(stats.get("control_wall_s", 0.0))
    deferred_stage = float(stats.get("deferred_stage_wall_s", 0.0))
    commit_wall = float(stats.get("commit_wall_s", 0.0))
    deferred_total = deferred_stage + commit_wall

    suffix_valid = bool(suffix) and all(
        row["calls_balanced"]
        and row["bytes_balanced"]
        and row["commit_bytes_exact"]
        for row in suffix.values()
    )
    valid = bool(
        stats.get("enabled") is True
        and control_calls > 0
        and control_calls == deferred_calls
        and control_bytes > 0
        and control_bytes == deferred_bytes
        and commit_bytes == deferred_bytes
        and int(stats.get("commit_calls", 0)) > 0
        and int(stats.get("committed_layers", 0)) > 0
        and suffix_valid
        and tensor_summary.get("tensor_consumer_attribution_valid") is True
        and loader.get("loader_attribution_valid") is True
    )

    speedup = (
        control_wall / deferred_total
        if valid and control_wall > 0 and deferred_total > 0
        else None
    )

    return {
        "schema": 1,
        "metadata_bulk_ab_valid": valid,
        "classification": (
            "METADATA_BULK_AB_POSITIVE"
            if valid and speedup is not None and speedup > 1.0
            else "METADATA_BULK_AB_NEGATIVE"
            if valid
            else "METADATA_BULK_AB_INVALID"
        ),
        "main_weights_s": loader["startup"]["main_weights_s"],
        "control": {
            "calls": control_calls,
            "bytes": control_bytes,
            "wall_s": control_wall,
            "ms_per_tensor": (
                1000.0 * control_wall / control_calls
                if control_calls
                else None
            ),
        },
        "deferred": {
            "calls": deferred_calls,
            "bytes": deferred_bytes,
            "stage_wall_s": deferred_stage,
            "bulk_commit_calls": int(stats.get("commit_calls", 0)),
            "bulk_commit_bytes": commit_bytes,
            "bulk_commit_wall_s": commit_wall,
            "committed_layers": int(stats.get("committed_layers", 0)),
            "strided_commit_calls": int(stats.get("strided_commit_calls", 0)),
            "index_commit_calls": int(stats.get("index_commit_calls", 0)),
            "total_wall_s": deferred_total,
            "ms_per_tensor_including_commit": (
                1000.0 * deferred_total / deferred_calls
                if deferred_calls
                else None
            ),
        },
        "comparison": {
            "calls_balanced": control_calls == deferred_calls and control_calls > 0,
            "bytes_balanced": control_bytes == deferred_bytes and control_bytes > 0,
            "commit_bytes_exact": commit_bytes == deferred_bytes and deferred_bytes > 0,
            "deferred_vs_control_speedup": speedup,
            "deferred_wall_ratio": (
                deferred_total / control_wall
                if valid and control_wall > 0
                else None
            ),
            "gpu_call_reduction": (
                deferred_calls / int(stats.get("commit_calls", 0))
                if int(stats.get("commit_calls", 0)) > 0
                else None
            ),
        },
        "by_suffix": suffix,
        "tensor_consumer_mixed_arm": {
            "main_weights_s": tensor_summary.get("main_weights_s"),
            "totals": tensor_summary.get("totals"),
            "by_suffix": tensor_summary.get("by_suffix"),
            "by_scope": tensor_summary.get("by_scope"),
            "by_size_bin": tensor_summary.get("by_size_bin"),
            "copy_reconciliation": tensor_summary.get(
                "exl3_copy_reconciliation"
            ),
        },
        "kernel_model_load_window": loader.get("kernel_model_load_window"),
        "interpretation_contract": {
            "arm_assignment": (
                "even local expert ids use existing tiny GPU writes; odd local "
                "expert ids defer mul1/mcg/suh/svh and bulk-commit at the end "
                "of that RoutedExperts.load_weights invocation"
            ),
            "total_deferred_wall": (
                "deferred per-tensor staging wall plus measured bulk commit wall"
            ),
            "main_weight_accounting": (
                "bulk commit runs before the layer load_weights generator "
                "returns, so it remains inside model load_weights timing rather "
                "than being moved to postload"
            ),
            "qualification": (
                "This is an interleaved mechanism A/B, not production "
                "qualification. A positive result justifies an all-expert bulk "
                "qualification boot."
            ),
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--trace", type=Path, required=True)
    ap.add_argument("--tensor-summary", type=Path, required=True)
    ap.add_argument("--loader-summary", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    out = summarize(
        _load_jsonl(args.trace),
        json.loads(args.tensor_summary.read_text()),
        json.loads(args.loader_summary.read_text()),
    )
    args.out.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    return 0 if out["metadata_bulk_ab_valid"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
