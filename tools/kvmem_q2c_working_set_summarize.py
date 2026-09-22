#!/usr/bin/env python3
"""Summarize original-QSA page demand for CPU-backed selection reload."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


PAGE_BYTES = 32768


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _percentile(values: list[int], fraction: float) -> int:
    if not values:
        return 0
    rows = sorted(values)
    return rows[math.ceil(fraction * len(rows)) - 1]


def summarize(rows: list[dict[str, Any]], plan: dict[str, Any]) -> dict[str, Any]:
    prompt_tokens = int(plan["query_span"][1])
    prompt_last = prompt_tokens - 1
    expected_layers = int(plan["expected_qsa_layers"])
    cap = int(plan["physical_page_count"])
    resident = int(plan["resident_page_count"])
    active = int(plan["active_reserve_pages"])
    records = [
        row for row in rows
        if row.get("mode") == "a_full_original"
        and int(row.get("last_pos", prompt_tokens)) <= prompt_last
        and not (
            int(row.get("query_rows", 0)) == 1
            and int(row.get("first_pos", -1)) == 0
            and int(row.get("last_pos", -1)) == 0
        )
    ]
    layers = sorted({str(row["layer"]) for row in records})
    required = (
        "unique_pages_before",
        "unique_historical_pages_before",
        "unique_nonresident_historical_pages_before",
        "unique_pages_with_current_writes_before",
    )
    fields_gate = bool(records) and all(
        all(field in row for field in required) for row in records
    )
    layer_gate = len(layers) == expected_layers

    working = [
        int(row["unique_pages_with_current_writes_before"])
        for row in records if fields_gate
    ]
    selected = [int(row["unique_pages_before"]) for row in records if fields_gate]
    misses = [
        int(row["unique_nonresident_historical_pages_before"])
        for row in records if fields_gate
    ]
    maximum = max(working, default=0)
    max_row = max(
        records,
        key=lambda row: int(row.get("unique_pages_with_current_writes_before", 0)),
        default={},
    )
    by_layer: dict[str, dict[str, Any]] = {}
    for layer in layers:
        layer_rows = [row for row in records if str(row["layer"]) == layer]
        values = [
            int(row["unique_pages_with_current_writes_before"])
            for row in layer_rows
        ] if fields_gate else []
        layer_misses = [
            int(row["unique_nonresident_historical_pages_before"])
            for row in layer_rows
        ] if fields_gate else []
        by_layer[layer] = {
            "chunks": len(layer_rows),
            "max_working_pages": max(values, default=0),
            "p95_working_pages": _percentile(values, 0.95),
            "cold_reload_pages_upper": sum(layer_misses),
            "cold_reload_gib_upper": sum(layer_misses) * PAGE_BYTES / 2**30,
        }

    logical_pages = math.ceil(prompt_tokens / int(plan["page_tokens"]))
    cpu_backing_bytes = logical_pages * PAGE_BYTES * expected_layers
    whole_chunk_feasible = bool(fields_gate and maximum <= cap)
    row_batch_sizes = (512, 256, 128, 64)
    row_batch_evidence: dict[str, dict[str, Any]] = {}
    for size in row_batch_sizes:
        field = f"row_batch_{size}_max_working_pages"
        values = [int(row[field]) for row in records if field in row]
        row_batch_evidence[str(size)] = {
            "records": len(values),
            "max_working_pages": max(values, default=0),
            "p95_working_pages": _percentile(values, 0.95),
            "events_over_cap": sum(value > cap for value in values),
            "feasible": bool(len(values) == len(records) and max(values, default=cap + 1) <= cap),
        }
    feasible_batches = [
        size for size in row_batch_sizes
        if row_batch_evidence[str(size)]["feasible"]
    ]
    largest_feasible_batch = 1024 if whole_chunk_feasible else (
        max(feasible_batches, default=0)
    )
    if whole_chunk_feasible:
        classification = "Q2C_RELOAD_WHOLE_CHUNK_FEASIBLE"
    elif largest_feasible_batch:
        classification = f"Q2C_RELOAD_ROW_BATCH_{largest_feasible_batch}_FEASIBLE"
    else:
        classification = "Q2C_RELOAD_SMALLER_QUERY_ROW_SPLIT_REQUIRED"
    return {
        "schema": 1,
        "classification": classification,
        "evidence_gate": bool(fields_gate and layer_gate),
        "mode": "a_full_original",
        "prompt_tokens": prompt_tokens,
        "records": len(records),
        "layers": layers,
        "layer_count": len(layers),
        "expected_layer_count": expected_layers,
        "physical_page_cap": cap,
        "frozen_resident_pages": resident,
        "active_reserve_pages": active,
        "frozen_contract_spare_pages": cap - resident - active,
        "max_selected_pages": max(selected, default=0),
        "max_working_pages_with_current_writes": maximum,
        "p50_working_pages_with_current_writes": _percentile(working, 0.50),
        "p95_working_pages_with_current_writes": _percentile(working, 0.95),
        "chunks_over_cap": sum(value > cap for value in working),
        "whole_1024_query_chunk_feasible": whole_chunk_feasible,
        "largest_measured_feasible_query_row_batch": largest_feasible_batch,
        "row_batch_evidence": row_batch_evidence,
        "max_event": {
            key: max_row.get(key)
            for key in (
                "layer", "first_pos", "last_pos", "query_rows",
                "unique_pages_before",
                "unique_historical_pages_before",
                "unique_nonresident_historical_pages_before",
                "unique_pages_with_current_writes_before",
            )
        },
        "cold_reload_pages_upper_all_events": sum(misses),
        "cold_reload_gib_upper_all_events": sum(misses) * PAGE_BYTES / 2**30,
        "full_history_cpu_backing_pages_per_layer": logical_pages,
        "full_history_cpu_backing_bytes_all_layers": cpu_backing_bytes,
        "full_history_cpu_backing_gib_all_layers": cpu_backing_bytes / 2**30,
        "by_layer": by_layer,
        "interpretation": {
            "cold_reload_upper_bound": (
                "Counts every non-frozen historical page selected by every chunk as a "
                "fresh transfer; a dynamic cache can reduce this through reuse."
            ),
            "capacity_contract": (
                "The current 4096 sticky + 64 active contract has zero spare pages. "
                "Selection-driven reload must make historical residency evictable or "
                "define a separate bounded workspace."
            ),
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stats", type=Path, required=True)
    ap.add_argument("--plan", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    result = summarize(load_jsonl(args.stats), json.loads(args.plan.read_text()))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    return 0 if result["evidence_gate"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
