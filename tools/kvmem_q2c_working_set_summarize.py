#!/usr/bin/env python3
"""Summarize original-QSA page demand for CPU-backed selection reload."""

from __future__ import annotations

import argparse
from collections import Counter
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
    chunk_tokens = int(plan["scheduler_chunk_tokens"])
    expected_spans = tuple(
        (start, min(start + chunk_tokens, prompt_tokens) - 1)
        for start in range(0, prompt_tokens, chunk_tokens)
    )
    prefill_rows = [
        row for row in rows
        if row.get("mode") == "a_full_original"
        and int(row.get("last_pos", prompt_tokens)) <= prompt_last
    ]
    expected_span_set = set(expected_spans)
    request_rows = [
        row for row in prefill_rows
        if (int(row.get("first_pos", -1)), int(row.get("last_pos", -1)))
        in expected_span_set
        and int(row.get("query_rows", 0))
        == int(row.get("last_pos", -1)) - int(row.get("first_pos", -1)) + 1
    ]
    layers = sorted({str(row["layer"]) for row in request_rows})
    key_counts = Counter(
        (
            str(row["layer"]),
            int(row["first_pos"]),
            int(row["last_pos"]),
        )
        for row in request_rows
    )
    expected_keys = {
        (layer, first, last)
        for layer in layers
        for first, last in expected_spans
    }
    observed_keys = set(key_counts)
    duplicate_keys = sum(count - 1 for count in key_counts.values() if count > 1)
    unexpected_prefill_records = len(prefill_rows) - len(request_rows)
    coverage_gate = bool(
        len(layers) == expected_layers
        and observed_keys == expected_keys
        and duplicate_keys == 0
        and unexpected_prefill_records == 0
    )
    records = request_rows
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
            "frozen_policy_nonresident_selected_pages_sum": sum(layer_misses),
            "frozen_policy_nonresident_selected_gib_sum": (
                sum(layer_misses) * PAGE_BYTES / 2**30
            ),
        }

    logical_pages = math.ceil(prompt_tokens / int(plan["page_tokens"]))
    cpu_backing_bytes = logical_pages * PAGE_BYTES * expected_layers
    whole_chunk_feasible = bool(fields_gate and coverage_gate and maximum <= cap)
    row_batch_sizes = (512, 256, 128, 64)
    row_batch_evidence: dict[str, dict[str, Any]] = {}
    for size in row_batch_sizes:
        field = f"row_batch_{size}_max_working_pages"
        selected_field = f"row_batch_{size}_max_selected_pages"
        cold_field = f"row_batch_{size}_cold_h2d_pages_no_reuse"
        subbatch_field = f"row_batch_{size}_subbatches"
        values = [int(row[field]) for row in records if field in row]
        selected_values = [
            int(row[selected_field]) for row in records if selected_field in row
        ]
        cold_values = [int(row[cold_field]) for row in records if cold_field in row]
        subbatch_values = [
            int(row[subbatch_field]) for row in records if subbatch_field in row
        ]
        complete = bool(
            coverage_gate
            and len(values) == len(records)
            and len(selected_values) == len(records)
            and len(cold_values) == len(records)
            and len(subbatch_values) == len(records)
        )
        row_batch_evidence[str(size)] = {
            "records": len(values),
            "complete": complete,
            "max_working_pages": max(values, default=0),
            "max_selected_pages": max(selected_values, default=0),
            "p95_working_pages": _percentile(values, 0.95),
            "events_over_cap": sum(value > cap for value in values),
            "subbatches": sum(subbatch_values),
            "cold_h2d_pages_no_reuse": sum(cold_values),
            "cold_h2d_gib_no_reuse": sum(cold_values) * PAGE_BYTES / 2**30,
            "feasible": bool(complete and max(values, default=cap + 1) <= cap),
        }
    feasible_batches = [
        size for size in row_batch_sizes
        if row_batch_evidence[str(size)]["feasible"]
    ]
    largest_feasible_batch = 1024 if whole_chunk_feasible else (
        max(feasible_batches, default=0)
    )
    evidence_gate = bool(fields_gate and layer_gate and coverage_gate)
    if not evidence_gate:
        classification = "Q2C_RELOAD_EVIDENCE_INCOMPLETE"
    elif whole_chunk_feasible:
        classification = "Q2C_RELOAD_WHOLE_CHUNK_FEASIBLE"
    elif largest_feasible_batch:
        classification = f"Q2C_RELOAD_ROW_BATCH_{largest_feasible_batch}_FEASIBLE"
    else:
        classification = "Q2C_RELOAD_SMALLER_QUERY_ROW_SPLIT_REQUIRED"
    return {
        "schema": 1,
        "classification": classification,
        "evidence_gate": evidence_gate,
        "mode": "a_full_original",
        "prompt_tokens": prompt_tokens,
        "scheduler_chunk_tokens": chunk_tokens,
        "records": len(records),
        "expected_records": expected_layers * len(expected_spans),
        "expected_chunks_per_layer": len(expected_spans),
        "coverage_gate": coverage_gate,
        "missing_records": len(expected_keys - observed_keys),
        "unexpected_records": len(observed_keys - expected_keys),
        "duplicate_records": duplicate_keys,
        "unexpected_prefill_records": unexpected_prefill_records,
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
        "frozen_policy_nonresident_selected_pages_sum": sum(misses),
        "frozen_policy_nonresident_selected_gib_sum": (
            sum(misses) * PAGE_BYTES / 2**30
        ),
        "full_history_cpu_backing_pages_per_layer": logical_pages,
        "full_history_cpu_backing_bytes_all_layers": cpu_backing_bytes,
        "full_history_cpu_backing_gib_all_layers": cpu_backing_bytes / 2**30,
        "by_layer": by_layer,
        "interpretation": {
            "row_batch_cold_h2d_no_reuse": (
                "For each query-row sub-batch, counts every unique processed "
                "historical selected page as a fresh H2D load. This is a valid "
                "cold/no-reuse H2D upper bound for that measured row split."
            ),
            "frozen_policy_metric": (
                "The nonresident-selected sum is relative to the old frozen plan. "
                "It is not a transfer bound for a dynamic replacement policy."
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
