#!/usr/bin/env python3
"""Summarize K1-Q2A physical-shadow resident-cache evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _load_jsonl(path: Path) -> list[dict]:
    out = []
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def summarize(response: dict, stats: list[dict], plan: dict) -> dict:
    layers = sorted({str(row["layer_name"]) for row in stats})
    expected_layers = int(plan["expected_qsa_layers"])
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
    attention_elements = sum(
        int(row["attention_elements"]) for row in stats
    )
    input_mapping_exact = bool(stats) and all(
        bool(row["input_mapping_exact"]) for row in stats
    )
    input_tokens_compared = sum(
        int(row["input_tokens_compared"]) for row in stats
    )
    first_bad_input_tokens = [
        int(row["first_bad_input_token"])
        for row in stats
        if row.get("first_bad_input_token") is not None
    ]
    bootstrap_pages = {
        str(row["layer_name"]): int(row["bootstrap_pages"]) for row in stats
    }
    bootstrap_ok = (
        len(bootstrap_pages) == expected_layers
        and all(
            value == int(plan["resident_page_count"])
            for value in bootstrap_pages.values()
        )
    )
    physical_pages = {int(row["resident_physical_pages"]) for row in stats}
    resident_page_tokens = {
        int(row["resident_page_tokens"]) for row in stats
    }
    full_block_tokens = {int(row["full_block_tokens"]) for row in stats}
    resident_table_widths = {
        int(row["resident_table_width"]) for row in stats
    }
    resident_cache_bytes = {
        int(row["resident_cache_bytes"]) for row in stats
    }
    physical_geometry_ok = (
        physical_pages == {int(plan["physical_page_count"])}
        and resident_page_tokens == {int(plan["page_tokens"])}
        and len(full_block_tokens) == 1
        and next(iter(full_block_tokens), 0) % int(plan["page_tokens"]) == 0
        and len(resident_table_widths) == 1
        and next(iter(resident_table_widths), 0) > int(plan["active_page0"])
        and len(resident_cache_bytes) == 1
    )

    hist_total = sum(int(row["historical_selected"]) for row in stats)
    hist_kept = sum(int(row["historical_resident_kept"]) for row in stats)
    dropped = sum(int(row["historical_selected_dropped"]) for row in stats)
    accounting_ok = hist_kept + dropped == hist_total
    exercised = dropped > 0
    layer_coverage_ok = len(layers) == expected_layers

    target_correct = bool(response.get("target_codes_in_order"))
    semantic_complete = response.get("finish_reason") == "stop"

    mapping_gate = bool(
        input_mapping_exact
        and bootstrap_ok
        and physical_geometry_ok
        and accounting_ok
        and exercised
        and layer_coverage_ok
    )
    semantic_gate = bool(target_correct and semantic_complete)

    if not mapping_gate:
        classification = "Q2A_INPUT_MAPPING_NO_GO"
        go = False
    elif not semantic_gate:
        classification = "Q2A_SEMANTIC_NO_GO"
        go = False
    elif attention_exact and max_abs == 0.0:
        classification = "Q2A_PHYSICAL_EXACT_GO"
        go = True
    else:
        classification = "Q2A_MAPPING_EXACT_SEMANTIC_GO_NONEXACT"
        go = True

    return {
        "schema": 1,
        "classification": classification,
        "physical_shadow_go": go,
        "mapping_gate": mapping_gate,
        "semantic_gate": semantic_gate,
        "target_correct": target_correct,
        "finish_reason": response.get("finish_reason"),
        "completion_tokens": int(
            (response.get("usage") or {}).get("completion_tokens", 0)
        ),
        "text": response.get("text"),
        "resident_budget_tokens": int(plan["budget_tokens"]),
        "resident_region_count": int(plan["resident_region_count"]),
        "resident_page_count": int(plan["resident_page_count"]),
        "active_reserve_pages": int(plan["active_reserve_pages"]),
        "physical_page_count": int(plan["physical_page_count"]),
        "evidence": {
            "records": len(stats),
            "layers": layers,
            "layer_count": len(layers),
            "expected_layer_count": expected_layers,
            "layer_coverage_ok": layer_coverage_ok,
            "attention_exact_all_records": attention_exact,
            "attention_max_abs": max_abs,
            "attention_mean_abs_max_record": mean_abs_max,
            "attention_mismatch_elements": mismatch_elements,
            "attention_elements": attention_elements,
            "attention_mismatch_fraction": (
                mismatch_elements / attention_elements
                if attention_elements else 0.0
            ),
            "input_mapping_exact_all_records": input_mapping_exact,
            "input_tokens_compared": input_tokens_compared,
            "first_bad_input_tokens": first_bad_input_tokens,
            "bootstrap_pages_by_layer": bootstrap_pages,
            "bootstrap_ok": bootstrap_ok,
            "physical_geometry_ok": physical_geometry_ok,
            "resident_page_tokens": sorted(resident_page_tokens),
            "full_block_tokens": sorted(full_block_tokens),
            "resident_table_widths": sorted(resident_table_widths),
            "resident_cache_bytes_per_layer": sorted(resident_cache_bytes),
            "resident_cache_mib_per_layer": (
                next(iter(resident_cache_bytes)) / (1024**2)
                if len(resident_cache_bytes) == 1 else None
            ),
            "resident_cache_gib_all_layers": (
                next(iter(resident_cache_bytes)) * expected_layers / (1024**3)
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
            "Q2A keeps the scheduler-owned full KV cache allocated and uses it "
            "as bootstrap/reference. The hard remap gate is byte-exact selected "
            "K/V input equivalence. Attention may be numerically non-exact when "
            "the same inputs execute through different PAGE_SIZE-specialized "
            "Triton kernels; that is classified separately and still requires "
            "the final target answer to remain correct. No GPU-memory reduction "
            "is claimed."
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
    return 0 if result["physical_shadow_go"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
