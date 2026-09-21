#!/usr/bin/env python3
"""Summarize K1-T transfer correctness and economics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def summarize(probe: dict, reference_gib_s: float) -> dict:
    if reference_gib_s <= 0:
        raise ValueError("reference_gib_s must be positive")

    stage = probe["transition"]
    geometry = probe["geometry"]
    h2d = probe["h2d_stage_in"]
    verify = probe["byte_verification"]
    table = probe["qsa_page_table"]
    backing = probe["cpu_backing"]

    gib = float(stage["stage_in_gib"])
    raw_floor_ms = gib / reference_gib_s * 1000.0
    event_ms = float(h2d["median_event_ms"])
    wall_ms = float(h2d["median_wall_ms"])

    event_ratio = event_ms / raw_floor_ms
    wall_ratio = wall_ms / raw_floor_ms
    if wall_ratio <= 1.5:
        perf_class = "CLOSE_TO_RAW_FLOOR"
    elif wall_ratio <= 2.5:
        perf_class = "MODERATE_TRANSFER_OVERHEAD"
    else:
        perf_class = "HIGH_TRANSFER_OVERHEAD"

    expected_stage_in_pages = (
        12 * int(geometry["pages_per_region"])
    )

    correctness_go = bool(
        verify["all_repeats_exact"]
        and table["all_stage_in_mappings_exact"]
        and table["all_evicted_pages_are_negative"]
        and backing["all_keys_hit_after_store"]
        and int(stage["stage_in_bytes"]) == 72 * 1024 * 1024
        and int(stage["query_replacements_regions"]) == 12
        and int(stage["stage_in_pages"]) == expected_stage_in_pages
    )

    return {
        "schema": 1,
        "transfer_correctness_go": correctness_go,
        "reference_raw_h2d_gib_s": reference_gib_s,
        "stage_in_gib": gib,
        "expected_stage_in_pages": expected_stage_in_pages,
        "raw_copy_floor_ms": raw_floor_ms,
        "median_worker_event_ms": event_ms,
        "median_submit_wait_wall_ms": wall_ms,
        "event_over_raw_floor": event_ratio,
        "wall_over_raw_floor": wall_ratio,
        "performance_class": perf_class,
        "note": (
            "Performance classification is diagnostic, not a hard K1-T gate. "
            "Correctness requires generic CPU backing, exact bytes and exact "
            "logical-page to physical-page residency mapping."
        ),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--probe", type=Path, required=True)
    ap.add_argument("--reference-gib-s", type=float, default=6.3494)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    result = summarize(
        json.loads(args.probe.read_text()),
        args.reference_gib_s,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    return 0 if result["transfer_correctness_go"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
