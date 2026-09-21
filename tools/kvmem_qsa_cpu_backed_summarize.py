#!/usr/bin/env python3
"""Summarize K1-Q2B real-model generic-CPU-backed resident-cache evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def summarize(response: dict, stats: list[dict], plan: dict) -> dict:
    layers = sorted({str(row["layer_name"]) for row in stats})
    expected_layers = int(plan["expected_qsa_layers"])

    by_layer = {}
    for row in stats:
        by_layer[str(row["layer_name"])] = row

    layer_coverage_ok = len(layers) == expected_layers
    bootstrap_pages = {
        name: int(row["bootstrap_pages"]) for name, row in by_layer.items()
    }
    bootstrap_ok = (
        len(bootstrap_pages) == expected_layers
        and all(
            value == int(plan["resident_page_count"])
            for value in bootstrap_pages.values()
        )
    )

    cpu_present = {
        name: bool(row["cpu_backing_all_present"])
        for name, row in by_layer.items()
    }
    bootstrap_exact = {
        name: bool(row["bootstrap_all_pages_exact"])
        for name, row in by_layer.items()
    }
    bootstrap_compared = {
        name: int(row["bootstrap_pages_compared"])
        for name, row in by_layer.items()
    }
    bad_bootstrap_pages = [
        int(row["first_bad_bootstrap_page"])
        for row in by_layer.values()
        if row.get("first_bad_bootstrap_page") is not None
    ]

    page_size_bytes = {
        int(row["transfer_page_size_bytes"]) for row in by_layer.values()
    }
    if len(page_size_bytes) == 1:
        expected_transfer_bytes = (
            int(plan["resident_page_count"]) * next(iter(page_size_bytes))
        )
    else:
        expected_transfer_bytes = None

    d2h_bytes = {
        name: int(row["d2h_publish_bytes"]) for name, row in by_layer.items()
    }
    h2d_bytes = {
        name: int(row["h2d_stage_in_bytes"]) for name, row in by_layer.items()
    }
    d2h_jobs = {
        name: int(row["d2h_publish_jobs"]) for name, row in by_layer.items()
    }
    d2h_event = {
        name: float(row["d2h_event_seconds"]) for name, row in by_layer.items()
    }
    d2h_wall = {
        name: float(row["d2h_wall_seconds"]) for name, row in by_layer.items()
    }
    h2d_event = {
        name: float(row["h2d_event_seconds"]) for name, row in by_layer.items()
    }
    h2d_wall = {
        name: float(row["h2d_wall_seconds"]) for name, row in by_layer.items()
    }
    transfer_tensor_bytes = {
        int(row["transfer_tensor_bytes"]) for row in by_layer.values()
    }
    expected_publish_jobs = (
        int(plan["resident_page_count"])
        + int(plan["publication_staging_pages"])
        - 1
    ) // int(plan["publication_staging_pages"])
    transfer_tensor_geometry_ok = bool(
        len(page_size_bytes) == 1
        and len(transfer_tensor_bytes) == 1
        and next(iter(transfer_tensor_bytes))
        == int(plan["transfer_tensor_page_count"]) * next(iter(page_size_bytes))
    )
    bootstrap_sources = {
        str(row["bootstrap_source"]) for row in by_layer.values()
    }

    transfer_gate = bool(
        layer_coverage_ok
        and len(cpu_present) == expected_layers
        and all(cpu_present.values())
        and len(bootstrap_exact) == expected_layers
        and all(bootstrap_exact.values())
        and not bad_bootstrap_pages
        and all(
            value == int(plan["resident_page_count"])
            for value in bootstrap_compared.values()
        )
        and expected_transfer_bytes is not None
        and all(value == expected_transfer_bytes for value in d2h_bytes.values())
        and all(value == expected_transfer_bytes for value in h2d_bytes.values())
        and len(d2h_jobs) == expected_layers
        and all(value == expected_publish_jobs for value in d2h_jobs.values())
        and transfer_tensor_geometry_ok
        and bootstrap_sources == {"vllm_generic_cpu_offload"}
    )

    input_mapping_exact = bool(stats) and all(
        bool(row["input_mapping_exact"]) for row in stats
    )
    input_tokens_compared = sum(
        int(row["input_tokens_compared"]) for row in stats
    )
    bad_input_tokens = [
        int(row["first_bad_input_token"])
        for row in stats
        if row.get("first_bad_input_token") is not None
    ]

    physical_pages = {int(row["resident_physical_pages"]) for row in stats}
    resident_page_tokens = {int(row["resident_page_tokens"]) for row in stats}
    full_block_tokens = {int(row["full_block_tokens"]) for row in stats}
    table_widths = {int(row["resident_table_width"]) for row in stats}
    resident_cache_bytes = {int(row["resident_cache_bytes"]) for row in stats}
    physical_geometry_ok = bool(
        physical_pages == {int(plan["physical_page_count"])}
        and resident_page_tokens == {int(plan["page_tokens"])}
        and len(full_block_tokens) == 1
        and next(iter(full_block_tokens), 0) % int(plan["page_tokens"]) == 0
        and len(table_widths) == 1
        and next(iter(table_widths), 0) > int(plan["active_page0"])
        and len(resident_cache_bytes) == 1
        and len(transfer_tensor_bytes) == 1
    )

    hist_total = sum(int(row["historical_selected"]) for row in stats)
    hist_kept = sum(int(row["historical_resident_kept"]) for row in stats)
    dropped = sum(int(row["historical_selected_dropped"]) for row in stats)
    accounting_ok = hist_kept + dropped == hist_total
    exercised = dropped > 0

    mapping_gate = bool(
        input_mapping_exact
        and physical_geometry_ok
        and bootstrap_ok
        and accounting_ok
        and exercised
        and layer_coverage_ok
    )

    attention_exact = bool(stats) and all(
        bool(row["attention_exact"]) for row in stats
    )
    max_abs = max(
        (float(row["attention_max_abs"]) for row in stats),
        default=float("inf"),
    )
    mean_abs_max = max(
        (float(row["attention_mean_abs"]) for row in stats),
        default=float("inf"),
    )
    mismatch_elements = sum(
        int(row["attention_mismatch_elements"]) for row in stats
    )
    attention_elements = sum(int(row["attention_elements"]) for row in stats)

    target_correct = bool(response.get("target_codes_in_order"))
    semantic_complete = response.get("finish_reason") == "stop"
    semantic_gate = bool(target_correct and semantic_complete)

    if not transfer_gate:
        classification = "Q2B_CPU_TRANSFER_NO_GO"
        go = False
    elif not mapping_gate:
        classification = "Q2B_INPUT_MAPPING_NO_GO"
        go = False
    elif not semantic_gate:
        classification = "Q2B_SEMANTIC_NO_GO"
        go = False
    elif attention_exact and max_abs == 0.0:
        classification = "Q2B_CPU_BACKED_EXACT_GO"
        go = True
    else:
        classification = "Q2B_CPU_BACKED_MAPPING_EXACT_SEMANTIC_GO_NONEXACT"
        go = True

    d2h_total_bytes = sum(d2h_bytes.values())
    h2d_total_bytes = sum(h2d_bytes.values())

    return {
        "schema": 1,
        "classification": classification,
        "cpu_backed_go": go,
        "transfer_gate": transfer_gate,
        "physical_mapping_go": mapping_gate,
        "semantic_go": semantic_gate,
        "target_correct": target_correct,
        "finish_reason": response.get("finish_reason"),
        "completion_tokens": int(
            (response.get("usage") or {}).get("completion_tokens", 0)
        ),
        "text": response.get("text"),
        "resident_budget_tokens": int(plan["budget_tokens"]),
        "resident_page_count": int(plan["resident_page_count"]),
        "active_reserve_pages": int(plan["active_reserve_pages"]),
        "physical_page_count": int(plan["physical_page_count"]),
        "publication_staging_pages": int(plan["publication_staging_pages"]),
        "transfer_tensor_page_count": int(plan["transfer_tensor_page_count"]),
        "transfer": {
            "cpu_backing_all_present_by_layer": cpu_present,
            "bootstrap_all_pages_exact_by_layer": bootstrap_exact,
            "bootstrap_pages_compared_by_layer": bootstrap_compared,
            "first_bad_bootstrap_pages": bad_bootstrap_pages,
            "page_size_bytes": sorted(page_size_bytes),
            "expected_bytes_per_layer": expected_transfer_bytes,
            "d2h_publish_bytes_by_layer": d2h_bytes,
            "h2d_stage_in_bytes_by_layer": h2d_bytes,
            "d2h_publish_jobs_by_layer": d2h_jobs,
            "expected_publish_jobs_per_layer": expected_publish_jobs,
            "transfer_tensor_geometry_ok": transfer_tensor_geometry_ok,
            "d2h_total_bytes": d2h_total_bytes,
            "h2d_total_bytes": h2d_total_bytes,
            "d2h_total_gib": d2h_total_bytes / (1024**3),
            "h2d_total_gib": h2d_total_bytes / (1024**3),
            "d2h_event_seconds_sum_layers": sum(d2h_event.values()),
            "d2h_wall_seconds_sum_layers": sum(d2h_wall.values()),
            "h2d_event_seconds_sum_layers": sum(h2d_event.values()),
            "h2d_wall_seconds_sum_layers": sum(h2d_wall.values()),
            "bootstrap_sources": sorted(bootstrap_sources),
            "transfer_tensor_bytes_per_layer": sorted(transfer_tensor_bytes),
            "transfer_tensor_mib_per_layer": (
                next(iter(transfer_tensor_bytes)) / (1024**2)
                if len(transfer_tensor_bytes) == 1 else None
            ),
        },
        "evidence": {
            "records": len(stats),
            "layers": layers,
            "layer_count": len(layers),
            "expected_layer_count": expected_layers,
            "layer_coverage_ok": layer_coverage_ok,
            "bootstrap_ok": bootstrap_ok,
            "physical_geometry_ok": physical_geometry_ok,
            "input_mapping_exact_all_records": input_mapping_exact,
            "input_tokens_compared": input_tokens_compared,
            "first_bad_input_tokens": bad_input_tokens,
            "attention_exact_all_records": attention_exact,
            "attention_max_abs": max_abs,
            "attention_mean_abs_max_record": mean_abs_max,
            "attention_mismatch_elements": mismatch_elements,
            "attention_elements": attention_elements,
            "attention_mismatch_fraction": (
                mismatch_elements / attention_elements
                if attention_elements else 0.0
            ),
            "resident_page_tokens": sorted(resident_page_tokens),
            "full_block_tokens": sorted(full_block_tokens),
            "resident_table_widths": sorted(table_widths),
            "resident_cache_bytes_per_layer": sorted(resident_cache_bytes),
            "resident_cache_mib_per_layer": (
                next(iter(resident_cache_bytes)) / (1024**2)
                if len(resident_cache_bytes) == 1 else None
            ),
            "cross_granularity_ratio": (
                next(iter(full_block_tokens)) / int(plan["page_tokens"])
                if len(full_block_tokens) == 1 else None
            ),
            "historical_selected": hist_total,
            "historical_resident_kept": hist_kept,
            "historical_selected_dropped": dropped,
            "historical_visibility_rate": (
                hist_kept / hist_total if hist_total else 1.0
            ),
            "accounting_ok": accounting_ok,
            "mask_exercised": exercised,
        },
        "note": (
            "Q2B proves the real model can publish the frozen historical "
            "resident set through vLLM generic CPU offload and stage it back "
            "into the independent 16-token resident cache while preserving "
            "byte-exact mapped K/V and final target semantics. The full "
            "scheduler KV remains allocated as a reference; no memory or "
            "production-TTFT claim is made."
        ),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--response", type=Path, required=True)
    ap.add_argument("--stats", type=Path, required=True)
    ap.add_argument("--plan", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    result = summarize(
        json.loads(args.response.read_text()),
        _load_jsonl(args.stats),
        json.loads(args.plan.read_text()),
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    return 0 if result["cpu_backed_go"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
