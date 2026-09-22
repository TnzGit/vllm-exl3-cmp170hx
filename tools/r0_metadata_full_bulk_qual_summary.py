#!/usr/bin/env python3
"""Summarize control/full/control production qualification for metadata bulk."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


WEIGHTS_RE = re.compile(r"Loading weights took\s+([0-9]+(?:\.[0-9]+)?)\s+seconds")


def _weight_times(path: Path) -> dict:
    vals = [float(x) for x in WEIGHTS_RE.findall(path.read_text(errors="replace"))]
    if not vals:
        raise ValueError(f"no weight-load timings in {path}")
    return {
        "main_weights_s": vals[0],
        "draft_weights_s": vals[1] if len(vals) > 1 else None,
        "total_weights_s": sum(vals[:2]) if len(vals) > 1 else vals[0],
        "all_weight_lines_s": vals,
    }


def _tokens(path: Path) -> list[str]:
    row = json.loads(path.read_text())
    pieces = row.get("token_pieces")
    if not pieces:
        pieces = (row.get("cells") or [{}])[0].get("token_pieces") or []
    out: list[str] = []
    for piece in pieces:
        if isinstance(piece, list):
            out.extend(str(x) for x in piece)
        else:
            out.append(str(piece))
    return out


def _final_trace(path: Path) -> dict:
    rows = [json.loads(x) for x in path.read_text().splitlines() if x.strip()]
    for row in rows:
        if row.get("tag") == "ALL_WEIGHTS_LOADED_BEFORE_POSTLOAD":
            return row
    raise ValueError("missing ALL_WEIGHTS_LOADED_BEFORE_POSTLOAD")


def summarize(
    control_a_log: Path,
    full_log: Path,
    control_b_log: Path,
    control_a_cell: Path,
    full_cell: Path,
    control_b_cell: Path,
    full_trace: Path,
    xid_json: Path,
    restore_json: Path,
    min_main_gain_pct: float = 1.0,
    max_control_drift_pct: float = 15.0,
) -> dict:
    a = _weight_times(control_a_log)
    f = _weight_times(full_log)
    b = _weight_times(control_b_log)
    main_ref = (a["main_weights_s"] + b["main_weights_s"]) / 2.0
    total_ref = (a["total_weights_s"] + b["total_weights_s"]) / 2.0
    control_drift_pct = (
        abs(a["main_weights_s"] - b["main_weights_s"]) / main_ref * 100.0
        if main_ref else float("inf")
    )
    main_gain_pct = (main_ref - f["main_weights_s"]) / main_ref * 100.0
    total_gain_pct = (total_ref - f["total_weights_s"]) / total_ref * 100.0

    ta, tf, tb = _tokens(control_a_cell), _tokens(full_cell), _tokens(control_b_cell)
    parity = bool(ta) and ta == tf == tb

    stats = (_final_trace(full_trace).get("metadata_bulk_ab") or {})
    bulk_accounting = bool(
        stats.get("mode") == "full"
        and stats.get("full_enabled") is True
        and int(stats.get("control_calls", 0)) == 0
        and int(stats.get("deferred_calls", 0)) > 0
        and int(stats.get("deferred_bytes", 0)) > 0
        and int(stats.get("commit_bytes", 0)) == int(stats.get("deferred_bytes", 0))
        and int(stats.get("commit_calls", 0)) > 0
        and int(stats.get("index_commit_calls", 0)) == 0
    )

    xid = json.loads(xid_json.read_text())
    restore = json.loads(restore_json.read_text())
    restore_ok = bool(
        restore.get("weight_utils_before")
        and restore.get("weight_utils_before") == restore.get("weight_utils_after")
        and restore.get("qsa_before")
        and restore.get("qsa_before") == restore.get("qsa_after")
    )
    hard_valid = bool(
        bulk_accounting
        and parity
        and int(xid.get("xid_delta", -1)) == 0
        and restore_ok
        and control_drift_pct <= max_control_drift_pct
    )
    performance_pass = bool(
        hard_valid
        and main_gain_pct >= min_main_gain_pct
        and total_gain_pct > 0.0
    )
    classification = (
        "METADATA_FULL_BULK_QUALIFIED"
        if performance_pass
        else "METADATA_FULL_BULK_VALID_BUT_NO_PERF_GATE"
        if hard_valid
        else "METADATA_FULL_BULK_INVALID"
    )

    return {
        "schema": 1,
        "classification": classification,
        "qualification_valid": hard_valid,
        "performance_gate_pass": performance_pass,
        "control_a": a,
        "full_bulk": f,
        "control_b": b,
        "comparison": {
            "control_reference_main_weights_s": main_ref,
            "control_reference_total_weights_s": total_ref,
            "control_main_drift_pct": control_drift_pct,
            "full_main_gain_pct": main_gain_pct,
            "full_total_weights_gain_pct": total_gain_pct,
            "min_main_gain_pct": min_main_gain_pct,
            "max_control_drift_pct": max_control_drift_pct,
        },
        "correctness": {
            "greedy_token_parity": parity,
            "token_count": len(tf),
            "xid_delta": int(xid.get("xid_delta", -1)),
            "restore_ok": restore_ok,
        },
        "bulk_accounting": {
            "valid": bulk_accounting,
            "mode": stats.get("mode"),
            "deferred_calls": int(stats.get("deferred_calls", 0)),
            "deferred_bytes": int(stats.get("deferred_bytes", 0)),
            "commit_calls": int(stats.get("commit_calls", 0)),
            "commit_bytes": int(stats.get("commit_bytes", 0)),
            "strided_commit_calls": int(stats.get("strided_commit_calls", 0)),
            "index_commit_calls": int(stats.get("index_commit_calls", 0)),
            "committed_layers": int(stats.get("committed_layers", 0)),
        },
        "interpretation_contract": {
            "performance": (
                "Qualification uses actual control/full/control weight-load wall; "
                "PR #29 projection is not a qualification denominator."
            ),
            "startup": (
                "Health and greedy request success are hard execution gates, but "
                "time-to-health is not a performance gate because compile/cache "
                "state is outside the metadata mechanism."
            ),
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--control-a-log", type=Path, required=True)
    ap.add_argument("--full-log", type=Path, required=True)
    ap.add_argument("--control-b-log", type=Path, required=True)
    ap.add_argument("--control-a-cell", type=Path, required=True)
    ap.add_argument("--full-cell", type=Path, required=True)
    ap.add_argument("--control-b-cell", type=Path, required=True)
    ap.add_argument("--full-trace", type=Path, required=True)
    ap.add_argument("--xid-json", type=Path, required=True)
    ap.add_argument("--restore-json", type=Path, required=True)
    ap.add_argument("--min-main-gain-pct", type=float, default=1.0)
    ap.add_argument("--max-control-drift-pct", type=float, default=15.0)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    out = summarize(
        args.control_a_log, args.full_log, args.control_b_log,
        args.control_a_cell, args.full_cell, args.control_b_cell,
        args.full_trace, args.xid_json, args.restore_json,
        args.min_main_gain_pct, args.max_control_drift_pct,
    )
    args.out.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    return 0 if out["qualification_valid"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
