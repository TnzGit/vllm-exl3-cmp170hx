#!/usr/bin/env python3
"""Summarize the Q2D full-source CPU reload correctness oracle."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from typing import Any


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def summarize(
    response: dict[str, Any], rows: list[dict[str, Any]], plan: dict[str, Any]
) -> dict[str, Any]:
    prompt_tokens = int(plan["query_span"][1])
    chunk_tokens = int(plan["scheduler_chunk_tokens"])
    expected_layers = int(plan["expected_qsa_layers"])
    cap = int(plan["physical_page_count"])
    page_tokens = int(plan["page_tokens"])
    expected_spans = tuple(
        (start, min(start + chunk_tokens, prompt_tokens) - 1)
        for start in range(0, prompt_tokens, chunk_tokens)
    )
    expected_span_set = set(expected_spans)
    events = [row for row in rows if row.get("event") == "q2d_reload_shadow"]
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
    duplicate_records = sum(count - 1 for count in counts.values() if count > 1)
    unexpected_prefill = [
        row for row in events
        if int(row.get("last_pos", prompt_tokens)) < prompt_tokens and row not in prefill
    ]
    coverage_gate = bool(
        len(layers) == expected_layers
        and observed_keys == expected_keys
        and duplicate_records == 0
        and not unexpected_prefill
    )
    required = (
        "max_working_pages",
        "physical_page_cap",
        "published_pages_total",
        "resident_slots",
        "peak_resident_slots",
        "stage_compare_exact",
        "attention_allclose",
        "attention_max_abs",
        "d2h_bytes_total",
        "h2d_bytes_total",
    )
    fields_gate = bool(events) and all(
        all(field in row for field in required) for row in events
    )
    capacity_gate = bool(
        fields_gate
        and all(
            int(row["physical_page_cap"]) == cap
            and int(row["max_working_pages"]) <= cap
            and int(row["resident_slots"]) <= cap
            and int(row["peak_resident_slots"]) <= cap
            for row in events
        )
    )
    stage_gate = bool(fields_gate and all(bool(row["stage_compare_exact"]) for row in events))
    attention_gate = bool(
        fields_gate
        and all(
            bool(row["attention_allclose"])
            and float(row["attention_max_abs"]) >= 0.0
            for row in events
        )
    )
    expected_prompt_pages = prompt_tokens // page_tokens
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
            int(row["published_pages_total"]) >= expected_prompt_pages
            for row in final_by_layer.values()
        )
    )
    semantic_gate = bool(
        response.get("target_codes_in_order")
        and response.get("finish_reason") == "stop"
        and int(response.get("usage", {}).get("prompt_tokens", -1)) == prompt_tokens
    )
    evidence_gate = bool(
        coverage_gate
        and fields_gate
        and capacity_gate
        and stage_gate
        and attention_gate
        and publication_gate
    )
    go = bool(evidence_gate and semantic_gate)
    if go:
        classification = "Q2D_CPU_RELOAD_SHADOW_SEMANTIC_GO_NONEXACT"
    elif not coverage_gate or not fields_gate:
        classification = "Q2D_CPU_RELOAD_SHADOW_EVIDENCE_INCOMPLETE"
    elif not capacity_gate:
        classification = "Q2D_CPU_RELOAD_SHADOW_CAPACITY_NO_GO"
    elif not stage_gate:
        classification = "Q2D_CPU_RELOAD_SHADOW_RESTORE_NO_GO"
    elif not attention_gate:
        classification = "Q2D_CPU_RELOAD_SHADOW_ATTENTION_NO_GO"
    else:
        classification = "Q2D_CPU_RELOAD_SHADOW_SEMANTIC_NO_GO"
    return {
        "schema": 1,
        "classification": classification,
        "reload_shadow_go": go,
        "diagnostic_full_source_shadow": True,
        "not_scheduler_ownership_qualification": True,
        "evidence_gate": evidence_gate,
        "semantic_gate": semantic_gate,
        "coverage_gate": coverage_gate,
        "fields_gate": fields_gate,
        "capacity_gate": capacity_gate,
        "stage_gate": stage_gate,
        "attention_gate": attention_gate,
        "publication_gate": publication_gate,
        "records": len(prefill),
        "all_events": len(events),
        "expected_records": expected_layers * len(expected_spans),
        "layer_count": len(layers),
        "missing_records": len(expected_keys - observed_keys),
        "unexpected_records": len(observed_keys - expected_keys),
        "duplicate_records": duplicate_records,
        "unexpected_prefill_records": len(unexpected_prefill),
        "expected_prompt_published_pages_per_layer": expected_prompt_pages,
        "min_final_published_pages_per_layer": min(
            (int(row["published_pages_total"]) for row in final_by_layer.values()),
            default=0,
        ),
        "physical_page_cap": cap,
        "max_working_pages": max(
            (int(row.get("max_working_pages", 0)) for row in events), default=0
        ),
        "max_peak_resident_slots": max(
            (int(row.get("peak_resident_slots", 0)) for row in events), default=0
        ),
        "split_calls": sum(int(row.get("split_calls", 0)) for row in events),
        "reload_miss_pages": sum(int(row.get("reload_miss_pages", 0)) for row in events),
        "selected_history_pages": sum(
            int(row.get("selected_history_pages", 0)) for row in events
        ),
        "max_attention_abs": max(
            (float(row.get("attention_max_abs", 0.0)) for row in events), default=0.0
        ),
        "attention_mismatch_elements": sum(
            int(row.get("attention_mismatch_elements", 0)) for row in events
        ),
        # Counters are cumulative per layer, so aggregate only each layer's
        # final observation rather than summing every event.
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
        load_jsonl(args.stats),
        json.loads(args.plan.read_text()),
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    return 0 if result["reload_shadow_go"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
