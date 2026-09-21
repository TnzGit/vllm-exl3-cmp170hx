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
    attention_exact = bool(stats) and all(bool(row["attention_exact"]) for row in stats)
    max_abs = max(
        (float(row["attention_max_abs"]) for row in stats),
        default=float("inf"),
    )
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
    go = bool(
        target_correct
        and semantic_complete
        and attention_exact
        and max_abs == 0.0
        and bootstrap_ok
        and physical_geometry_ok
        and accounting_ok
        and exercised
        and layer_coverage_ok
    )

    return {
        "schema": 1,
        "classification": (
            "Q2A_PHYSICAL_EXACT_GO" if go else "Q2A_PHYSICAL_NO_GO"
        ),
        "physical_shadow_go": go,
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
            "as bootstrap/reference. GO proves physical resident-page remapping "
            "and active-suffix writes, not GPU-memory reduction."
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
