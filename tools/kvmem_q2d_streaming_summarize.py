#!/usr/bin/env python3
"""Qualify scheduler-owned Q2D CPU-authoritative streaming runtime."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
from typing import Any


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _prefill_scheduler_rows(
    scheduler_rows: list[dict[str, Any]],
    prompt_tokens: int,
    page_tokens: int,
) -> tuple[list[dict[str, Any]], bool]:
    """Select the frozen prefill policy, excluding output-length-dependent decode."""
    prompt_pages = math.ceil(prompt_tokens / page_tokens)
    published_pages = prompt_tokens // page_tokens
    selected: list[dict[str, Any]] = []
    boundary_clean = True
    for row in scheduler_rows:
        event = row.get("event")
        if event == "q2d_scheduler_assign":
            pages = [int(page) for page in row.get("logical_pages", [])]
            if any(page < prompt_pages for page in pages) and any(
                page >= prompt_pages for page in pages
            ):
                boundary_clean = False
            if pages and all(page < prompt_pages for page in pages):
                selected.append(row)
        elif event == "q2d_scheduler_reclaim":
            pages = [int(page) for page in row.get("freed_logical_pages", [])]
            if any(page < published_pages for page in pages) and any(
                page >= published_pages for page in pages
            ):
                boundary_clean = False
            if pages:
                if all(page < published_pages for page in pages):
                    selected.append(row)
            elif int(row.get("processed_computed_tokens", 0)) <= prompt_tokens:
                # Compatibility for older synthetic fixtures without page IDs.
                selected.append(row)
    return selected, boundary_clean


def summarize(
    response: dict[str, Any],
    worker_rows: list[dict[str, Any]],
    scheduler_rows: list[dict[str, Any]],
    plan: dict[str, Any],
    trace_summary: dict[str, Any] | None = None,
    expected_io_mode: str = "staged_copy",
    expected_trace_sha256: str | None = None,
    expected_scheduler_digest: str | None = None,
    require_dynamic_table_oracle: bool = False,
) -> dict[str, Any]:
    if expected_io_mode not in ("staged_copy", "direct_dedicated_slots"):
        raise ValueError(f"unsupported expected_io_mode: {expected_io_mode}")
    prompt_tokens = int(plan["query_span"][1])
    chunk_tokens = int(plan["scheduler_chunk_tokens"])
    expected_layers = int(plan["expected_qsa_layers"])
    page_tokens = int(plan["page_tokens"])
    cap = int(plan["physical_page_count"])
    write_cap = int(plan["write_page_count"])
    read_cap = int(plan["read_cache_page_count"])
    expected_spans = tuple(
        (start, min(start + chunk_tokens, prompt_tokens) - 1)
        for start in range(0, prompt_tokens, chunk_tokens)
    )
    expected_span_set = set(expected_spans)
    events = [
        row for row in worker_rows
        if row.get("event") == "q2d_streaming_runtime"
    ]
    prefill = [
        row for row in events
        if (int(row.get("first_pos", -1)), int(row.get("last_pos", -1)))
        in expected_span_set
        and int(row.get("query_rows", 0))
        == int(row.get("last_pos", -1)) - int(row.get("first_pos", -1)) + 1
    ]
    layers = sorted({str(row["layer"]) for row in prefill})
    counts = Counter(
        (str(row["layer"]), int(row["first_pos"]), int(row["last_pos"]))
        for row in prefill
    )
    expected_keys = {
        (layer, first, last)
        for layer in layers
        for first, last in expected_spans
    }
    observed_keys = set(counts)
    duplicates = sum(count - 1 for count in counts.values() if count > 1)
    unexpected_prefill = [
        row for row in events
        if int(row.get("last_pos", prompt_tokens)) < prompt_tokens and row not in prefill
    ]
    coverage_gate = bool(
        len(layers) == expected_layers
        and observed_keys == expected_keys
        and duplicates == 0
        and not unexpected_prefill
    )
    required = (
        "current_write_pages",
        "resident_read_pages",
        "combined_real_pages",
        "max_working_pages",
        "physical_page_cap",
        "published_pages_total",
        "cpu_roundtrip_pages_total",
        "cpu_roundtrip_exact",
        "direct_load_verified_pages_total",
        "direct_consumer_syncs_total",
        "d2h_bytes_total",
        "h2d_bytes_total",
        "read_table_mode",
        "io_mode",
        "write_partition",
        "read_partition",
        "forward_exposed_wall_seconds",
        "write_mapping_wall_seconds",
        "kv_update_submit_wall_seconds",
        "publish_wall_seconds",
        "selection_plan_wall_seconds",
        "stage_history_wall_seconds",
        "table_build_wall_seconds",
        "attention_submit_wall_seconds",
        "d2h_event_seconds",
        "d2h_wall_seconds",
        "d2h_prepare_seconds",
        "d2h_submit_seconds",
        "d2h_wait_seconds",
        "d2h_finish_seconds",
        "h2d_event_seconds",
        "h2d_wall_seconds",
        "h2d_prepare_seconds",
        "h2d_submit_seconds",
        "h2d_wait_seconds",
        "h2d_finish_seconds",
        "d2d_copy_submit_seconds",
        "direct_consumer_sync_seconds",
        "publication_oracle_gather_submit_seconds",
        "stage_residency_lookup_seconds",
        "stage_slot_assignment_seconds",
        "stage_transfer_call_seconds",
        "stage_table_delta_seconds",
        "stage_touch_seconds",
        "trace_wall_seconds",
        "trace_records_total",
        "trace_bytes_total",
        "trace_truncated",
    )
    fields_gate = bool(events) and all(
        all(field in row for field in required) for row in events
    )
    timing_fields = tuple(field for field in required if field.endswith("_seconds"))
    timing_gate = bool(
        fields_gate
        and all(
            math.isfinite(float(row[field])) and float(row[field]) >= 0.0
            for row in events
            for field in timing_fields
        )
    )
    io_mode_gate = bool(
        fields_gate
        and all(row["io_mode"] == expected_io_mode for row in events)
        and (
            expected_io_mode != "direct_dedicated_slots"
            or (
                all(float(row["d2d_copy_submit_seconds"]) == 0.0 for row in events)
                and all(
                    int(row["direct_consumer_syncs_total"]) > 0
                    for row in events
                )
            )
        )
    )
    capacity_gate = bool(
        fields_gate
        and all(
            int(row["current_write_pages"]) <= write_cap
            and int(row["resident_read_pages"]) <= read_cap
            and int(row["combined_real_pages"]) <= cap
            and int(row["max_working_pages"]) <= cap
            and int(row["physical_page_cap"]) == cap
            for row in events
        )
    )
    mapping_gate = bool(
        fields_gate
        and all(
            row["read_table_mode"]
            == "dynamic_cpu_history_plus_scheduler_writes"
            and row["write_partition"] == [0, write_cap - 1]
            and row["read_partition"] == [write_cap, cap - 1]
            for row in events
        )
    )
    cpu_gate = bool(
        fields_gate and all(bool(row["cpu_roundtrip_exact"]) for row in events)
    )
    expected_published = prompt_tokens // page_tokens
    final_by_layer = {
        layer: max(
            (row for row in events if str(row["layer"]) == layer),
            key=lambda row: (int(row["last_pos"]), int(row["published_pages_total"])),
        )
        for layer in layers
    }
    publication_gate = bool(
        len(final_by_layer) == expected_layers
        and all(
            int(row["published_pages_total"]) >= expected_published
            and int(row["cpu_roundtrip_pages_total"]) >= expected_published
            for row in final_by_layer.values()
        )
    )
    expected_oracle_checks_by_layer = {
        layer: sum(
            int(row["split_calls"])
            for row in events
            if str(row["layer"]) == layer
        )
        for layer in layers
    }
    dynamic_table_oracle_gate = bool(
        not require_dynamic_table_oracle
        or (
            len(final_by_layer) == expected_layers
            and all(bool(row.get("dynamic_table_oracle_enabled")) for row in events)
            and all(
                bool(row.get("dynamic_table_oracle_enabled"))
                and int(row.get("dynamic_table_oracle_checks_total", -1))
                == expected_oracle_checks_by_layer[layer]
                and int(row.get("dynamic_table_oracle_pages_total", 0)) > 0
                for layer, row in final_by_layer.items()
            )
        )
    )
    assignments = [
        row for row in scheduler_rows if row.get("event") == "q2d_scheduler_assign"
    ]
    reclaims = [
        row for row in scheduler_rows if row.get("event") == "q2d_scheduler_reclaim"
    ]
    scheduler_gate = bool(
        assignments
        and reclaims
        and all(
            int(row.get("real_write_pages", write_cap + 1)) <= write_cap
            and int(row.get("peak_real_write_pages", write_cap + 1)) <= write_cap
            and all(0 <= int(x) < write_cap for x in row.get("write_ids", []))
            for row in assignments
        )
        and all(
            int(row.get("real_write_pages", write_cap + 1)) <= write_cap
            and int(row.get("peak_real_write_pages", write_cap + 1)) <= write_cap
            and all(0 <= int(x) < write_cap for x in row.get("freed_write_ids", []))
            for row in reclaims
        )
        and sum(len(row.get("logical_pages", [])) for row in assignments)
        >= (prompt_tokens + page_tokens - 1) // page_tokens
        and sum(int(row.get("freed_pages", 0)) for row in reclaims)
        >= expected_published
    )
    normalized_scheduler_rows = [
        {key: value for key, value in row.items() if key != "request_id"}
        for row in scheduler_rows
    ]
    scheduler_full_request_digest = hashlib.sha256(
        json.dumps(
            normalized_scheduler_rows,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    prefill_scheduler_rows, scheduler_prefill_boundary_gate = (
        _prefill_scheduler_rows(scheduler_rows, prompt_tokens, page_tokens)
    )
    normalized_prefill_scheduler_rows = [
        {key: value for key, value in row.items() if key != "request_id"}
        for row in prefill_scheduler_rows
    ]
    scheduler_digest = hashlib.sha256(
        json.dumps(
            normalized_prefill_scheduler_rows,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    memory_rows = [
        row for row in worker_rows if row.get("event") == "q2e_memory_census"
    ]
    memory_counts = Counter(str(row.get("layer")) for row in memory_rows)
    memory_gate = bool(
        len(memory_rows) == expected_layers
        and set(memory_counts) == set(layers)
        and all(count == 1 for count in memory_counts.values())
        and all(
            int(row.get("page_bytes", 0)) == 32768
            and row.get("io_mode") == expected_io_mode
            and row.get("backing_tensor") == (
                "dedicated_qsa_cache"
                if expected_io_mode == "direct_dedicated_slots"
                else "staging"
            )
            and row.get("staging_role") == (
                "allocated_unused"
                if expected_io_mode == "direct_dedicated_slots"
                else "transfer_bounce"
            )
            and int(row.get("addressable_pages", 0)) == cap
            and int(row.get("write_pages", 0)) == write_cap
            and int(row.get("read_pages", 0)) == read_cap
            and int(row.get("null_pages", 0)) == 1
            and int(row.get("staging_pages", 0)) == int(plan["staging_pages"])
            and int(row.get("cpu_backing_pages", 0)) == int(plan["cpu_page_count"])
            and int(row.get("dedicated_tensor_bytes", 0))
            == (cap + 1) * 32768
            and int(row.get("staging_tensor_bytes", 0))
            == int(plan["staging_pages"]) * 32768
            and int(row.get("cpu_backing_logical_bytes", 0))
            == int(plan["cpu_page_count"]) * 32768
            and int(row.get("dynamic_table_bytes", 0))
            == int(plan["cpu_page_count"]) * 4
            and int(row.get("selection_page_bitmap_bytes", 0))
            == int(plan["cpu_page_count"])
            for row in memory_rows
        )
    )
    expected_trace_records = sum(int(row["split_calls"]) for row in events) + sum(
        (int(row["published_pages_this_forward"]) + int(plan["staging_pages"]) - 1)
        // int(plan["staging_pages"])
        for row in events
    )
    expected_trace_history_pages = sum(
        int(row["selected_history_pages"]) for row in events
    )
    expected_trace_missing_pages = sum(
        int(row["reload_miss_pages"]) + int(row["published_pages_this_forward"])
        for row in events
    )
    expected_trace_by_layer = {
        str(next(
            row["layer_id"] for row in memory_rows if str(row["layer"]) == layer
        )): sum(
            int(row["split_calls"])
            + (
                int(row["published_pages_this_forward"])
                + int(plan["staging_pages"])
                - 1
            ) // int(plan["staging_pages"])
            for row in events
            if str(row["layer"]) == layer
        )
        for layer in layers
    } if memory_gate else {}
    trace_gate = trace_summary is None or bool(
        trace_summary.get("complete")
        and not trace_summary.get("truncated")
        and int(trace_summary.get("records", -1)) == expected_trace_records
        and int(trace_summary.get("history_pages", -1))
        == expected_trace_history_pages
        and int(trace_summary.get("missing_pages", -1))
        == expected_trace_missing_pages
        and int(trace_summary.get("assigned_slots", -1))
        == expected_trace_missing_pages
        and int(trace_summary.get("max_query_rows", -1))
        <= int(plan["query_row_batch"])
        and int(trace_summary.get("max_history_pages", -1)) <= read_cap
        and trace_summary.get("per_layer_records") == expected_trace_by_layer
    )
    paired_trace_exact = bool(
        expected_trace_sha256 is None
        or (
            trace_summary is not None
            and trace_summary.get("sha256") == expected_trace_sha256
        )
    )
    paired_policy_gate = bool(
        scheduler_prefill_boundary_gate
        and (
            expected_scheduler_digest is None
            or scheduler_digest == expected_scheduler_digest
        )
    )
    semantic_gate = bool(
        response.get("target_codes_in_order")
        and response.get("finish_reason") == "stop"
        and int(response.get("usage", {}).get("prompt_tokens", -1)) == prompt_tokens
    )
    evidence_gate = bool(
        coverage_gate and fields_gate and timing_gate and io_mode_gate
        and capacity_gate and mapping_gate
        and cpu_gate and publication_gate and scheduler_gate and memory_gate
        and trace_gate and paired_policy_gate and dynamic_table_oracle_gate
    )
    go = bool(evidence_gate and semantic_gate)
    if go:
        classification = "Q2D_CPU_AUTHORITATIVE_STREAMING_SEMANTIC_GO"
    elif (
        not coverage_gate or not fields_gate or not timing_gate or not io_mode_gate
        or not memory_gate or not trace_gate or not paired_policy_gate
        or not dynamic_table_oracle_gate
    ):
        classification = "Q2D_STREAMING_EVIDENCE_INCOMPLETE"
    elif not scheduler_gate or not capacity_gate:
        classification = "Q2D_STREAMING_OWNERSHIP_NO_GO"
    elif not cpu_gate or not publication_gate:
        classification = "Q2D_STREAMING_CPU_AUTHORITY_NO_GO"
    elif not mapping_gate:
        classification = "Q2D_STREAMING_READ_WRITE_MAPPING_NO_GO"
    else:
        classification = "Q2D_STREAMING_SEMANTIC_NO_GO"
    return {
        "schema": 1,
        "classification": classification,
        "streaming_qualification_go": go,
        "evidence_gate": evidence_gate,
        "semantic_gate": semantic_gate,
        "coverage_gate": coverage_gate,
        "fields_gate": fields_gate,
        "timing_gate": timing_gate,
        "io_mode_gate": io_mode_gate,
        "io_mode": expected_io_mode,
        "capacity_gate": capacity_gate,
        "mapping_gate": mapping_gate,
        "cpu_authority_gate": cpu_gate,
        "publication_gate": publication_gate,
        "dynamic_table_oracle_gate": dynamic_table_oracle_gate,
        "dynamic_table_oracle_required": require_dynamic_table_oracle,
        "scheduler_gate": scheduler_gate,
        "memory_census_gate": memory_gate,
        "trace_gate": trace_gate,
        "paired_trace_exact": paired_trace_exact,
        "paired_policy_gate": paired_policy_gate,
        "scheduler_prefill_boundary_gate": scheduler_prefill_boundary_gate,
        "trace_enabled": trace_summary is not None,
        "records": len(prefill),
        "all_worker_events": len(events),
        "expected_records": expected_layers * len(expected_spans),
        "layer_count": len(layers),
        "missing_records": len(expected_keys - observed_keys),
        "unexpected_records": len(observed_keys - expected_keys),
        "duplicate_records": duplicates,
        "unexpected_prefill_records": len(unexpected_prefill),
        "physical_page_cap": cap,
        "write_page_cap": write_cap,
        "read_cache_page_cap": read_cap,
        "max_current_write_pages": max(
            (int(row.get("current_write_pages", 0)) for row in events), default=0
        ),
        "max_resident_read_pages": max(
            (int(row.get("resident_read_pages", 0)) for row in events), default=0
        ),
        "max_combined_real_pages": max(
            (int(row.get("combined_real_pages", 0)) for row in events), default=0
        ),
        "max_working_pages": max(
            (int(row.get("max_working_pages", 0)) for row in events), default=0
        ),
        "expected_prompt_published_pages_per_layer": expected_published,
        "min_final_published_pages_per_layer": min(
            (int(row["published_pages_total"]) for row in final_by_layer.values()),
            default=0,
        ),
        "min_final_roundtrip_pages_per_layer": min(
            (int(row["cpu_roundtrip_pages_total"]) for row in final_by_layer.values()),
            default=0,
        ),
        "min_final_direct_load_verified_pages_per_layer": min(
            (
                int(row["direct_load_verified_pages_total"])
                for row in final_by_layer.values()
            ),
            default=0,
        ),
        "min_final_direct_consumer_syncs_per_layer": min(
            (
                int(row["direct_consumer_syncs_total"])
                for row in final_by_layer.values()
            ),
            default=0,
        ),
        "min_final_dynamic_table_oracle_checks_per_layer": min(
            (
                int(row.get("dynamic_table_oracle_checks_total", 0))
                for row in final_by_layer.values()
            ),
            default=0,
        ),
        "min_expected_dynamic_table_oracle_checks_per_layer": min(
            expected_oracle_checks_by_layer.values(), default=0
        ),
        "dynamic_table_oracle_check_mismatches": sum(
            int(final_by_layer[layer].get("dynamic_table_oracle_checks_total", -1))
            != expected
            for layer, expected in expected_oracle_checks_by_layer.items()
        ),
        "min_final_dynamic_table_oracle_pages_per_layer": min(
            (
                int(row.get("dynamic_table_oracle_pages_total", 0))
                for row in final_by_layer.values()
            ),
            default=0,
        ),
        "scheduler_assign_events": len(assignments),
        "scheduler_reclaim_events": len(reclaims),
        "scheduler_assigned_pages": sum(
            len(row.get("logical_pages", [])) for row in assignments
        ),
        "scheduler_reclaimed_pages": sum(
            int(row.get("freed_pages", 0)) for row in reclaims
        ),
        "scheduler_peak_write_pages": max(
            (int(row.get("peak_real_write_pages", 0)) for row in assignments + reclaims),
            default=0,
        ),
        "reload_miss_pages": sum(
            int(row.get("reload_miss_pages", 0)) for row in events
        ),
        "selected_history_pages": sum(
            int(row.get("selected_history_pages", 0)) for row in events
        ),
        "d2h_bytes": sum(
            int(row["d2h_bytes_total"]) for row in final_by_layer.values()
        ),
        "h2d_bytes": sum(
            int(row["h2d_bytes_total"]) for row in final_by_layer.values()
        ),
        "d2h_jobs": sum(
            int(row.get("d2h_jobs_total", 0)) for row in final_by_layer.values()
        ),
        "h2d_jobs": sum(
            int(row.get("h2d_jobs_total", 0)) for row in final_by_layer.values()
        ),
        "timing_all_events_seconds": {
            field: sum(float(row[field]) for row in events)
            for field in timing_fields
        },
        "timing_prefill_seconds": {
            field: sum(float(row[field]) for row in prefill)
            for field in timing_fields
        },
        "memory_census": {
            "layers": len(memory_rows),
            "dedicated_tensor_bytes": sum(
                int(row.get("dedicated_tensor_bytes", 0)) for row in memory_rows
            ),
            "staging_tensor_bytes": sum(
                int(row.get("staging_tensor_bytes", 0)) for row in memory_rows
            ),
            "cpu_backing_logical_bytes": sum(
                int(row.get("cpu_backing_logical_bytes", 0)) for row in memory_rows
            ),
            "selection_page_bitmap_bytes": sum(
                int(row.get("selection_page_bitmap_bytes", 0))
                for row in memory_rows
            ),
            "max_cuda_memory_allocated_bytes": max(
                (int(row.get("cuda_memory_allocated_bytes", 0)) for row in memory_rows),
                default=0,
            ),
            "max_cuda_memory_reserved_bytes": max(
                (int(row.get("cuda_memory_reserved_bytes", 0)) for row in memory_rows),
                default=0,
            ),
        },
        "trace_expected_records": expected_trace_records,
        "trace_expected_history_pages": expected_trace_history_pages,
        "trace_expected_missing_pages": expected_trace_missing_pages,
        "trace_summary": trace_summary,
        "scheduler_policy_digest": scheduler_digest,
        "scheduler_full_request_digest": scheduler_full_request_digest,
        "scheduler_prefill_policy_rows": len(prefill_scheduler_rows),
        "expected_trace_sha256": expected_trace_sha256,
        "expected_scheduler_policy_digest": expected_scheduler_digest,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--response", type=Path, required=True)
    ap.add_argument("--worker-stats", type=Path, required=True)
    ap.add_argument("--scheduler-stats", type=Path, required=True)
    ap.add_argument("--plan", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--trace-summary", type=Path)
    ap.add_argument(
        "--expected-io-mode",
        choices=("staged_copy", "direct_dedicated_slots"),
        default="staged_copy",
    )
    ap.add_argument("--expected-trace-sha256")
    ap.add_argument("--expected-scheduler-digest")
    ap.add_argument("--require-dynamic-table-oracle", action="store_true")
    args = ap.parse_args()
    result = summarize(
        json.loads(args.response.read_text()),
        load_jsonl(args.worker_stats),
        load_jsonl(args.scheduler_stats),
        json.loads(args.plan.read_text()),
        json.loads(args.trace_summary.read_text()) if args.trace_summary else None,
        args.expected_io_mode,
        args.expected_trace_sha256,
        args.expected_scheduler_digest,
        args.require_dynamic_table_oracle,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    return 0 if result["streaming_qualification_go"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
