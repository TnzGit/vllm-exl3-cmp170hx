#!/usr/bin/env python3
"""Qualify same-forward full-KV QSA query-row splitting."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
from typing import Any


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def summarize(
    response: dict[str, Any],
    rows: list[dict[str, Any]],
    plan: dict[str, Any],
    split_rows: int,
) -> dict[str, Any]:
    prompt_tokens = int(plan["query_span"][1])
    chunk_tokens = int(plan["scheduler_chunk_tokens"])
    expected_layers = int(plan["expected_qsa_layers"])
    expected_spans = tuple(
        (start, min(start + chunk_tokens, prompt_tokens) - 1)
        for start in range(0, prompt_tokens, chunk_tokens)
    )
    expected_span_set = set(expected_spans)
    prefill = [
        row for row in rows
        if row.get("mode") == "d_full_split"
        and int(row.get("last_pos", prompt_tokens)) < prompt_tokens
    ]
    records = [
        row for row in prefill
        if (int(row.get("first_pos", -1)), int(row.get("last_pos", -1)))
        in expected_span_set
        and int(row.get("query_rows", 0))
        == int(row.get("last_pos", -1)) - int(row.get("first_pos", -1)) + 1
    ]
    layers = sorted({str(row["layer"]) for row in records})
    counts = Counter(
        (str(row["layer"]), int(row["first_pos"]), int(row["last_pos"]))
        for row in records
    )
    expected_keys = {
        (layer, first, last)
        for layer in layers
        for first, last in expected_spans
    }
    observed_keys = set(counts)
    duplicate_records = sum(count - 1 for count in counts.values() if count > 1)
    unexpected_prefill_records = len(prefill) - len(records)
    coverage_gate = bool(
        len(layers) == expected_layers
        and observed_keys == expected_keys
        and duplicate_records == 0
        and unexpected_prefill_records == 0
    )

    required = (
        "split_rows",
        "split_calls",
        "split_exact",
        "split_elements",
        "split_mismatch_elements",
        "split_max_abs",
    )
    fields_gate = bool(records) and all(
        all(field in row for field in required) for row in records
    )
    row_size_gate = bool(
        fields_gate and all(int(row["split_rows"]) == split_rows for row in records)
    )
    call_gate = bool(
        fields_gate
        and all(
            int(row["split_calls"])
            == math.ceil(int(row["query_rows"]) / split_rows)
            for row in records
        )
    )
    exact_gate = bool(
        fields_gate
        and all(
            bool(row["split_exact"])
            and int(row["split_mismatch_elements"]) == 0
            and float(row["split_max_abs"]) == 0.0
            and int(row["split_elements"]) > 0
            for row in records
        )
    )
    semantic_gate = bool(
        response.get("target_codes_in_order")
        and response.get("finish_reason") == "stop"
        and int(response.get("usage", {}).get("prompt_tokens", -1)) == prompt_tokens
    )
    evidence_gate = bool(
        coverage_gate and fields_gate and row_size_gate and call_gate and exact_gate
    )
    if evidence_gate and semantic_gate:
        classification = "Q2C_FULL_KV_ROW_SPLIT_EXACT_SEMANTIC_GO"
    elif not coverage_gate or not fields_gate:
        classification = "Q2C_FULL_KV_ROW_SPLIT_EVIDENCE_INCOMPLETE"
    elif not exact_gate:
        classification = "Q2C_FULL_KV_ROW_SPLIT_ATTENTION_NO_GO"
    else:
        classification = "Q2C_FULL_KV_ROW_SPLIT_SEMANTIC_NO_GO"
    return {
        "schema": 1,
        "classification": classification,
        "split_qualification_go": bool(evidence_gate and semantic_gate),
        "evidence_gate": evidence_gate,
        "semantic_gate": semantic_gate,
        "coverage_gate": coverage_gate,
        "fields_gate": fields_gate,
        "row_size_gate": row_size_gate,
        "call_gate": call_gate,
        "exact_gate": exact_gate,
        "split_rows": split_rows,
        "prompt_tokens": prompt_tokens,
        "records": len(records),
        "expected_records": expected_layers * len(expected_spans),
        "layer_count": len(layers),
        "expected_layer_count": expected_layers,
        "missing_records": len(expected_keys - observed_keys),
        "unexpected_records": len(observed_keys - expected_keys),
        "duplicate_records": duplicate_records,
        "unexpected_prefill_records": unexpected_prefill_records,
        "split_calls": sum(int(row.get("split_calls", 0)) for row in records),
        "split_elements": sum(int(row.get("split_elements", 0)) for row in records),
        "split_mismatch_elements": sum(
            int(row.get("split_mismatch_elements", 0)) for row in records
        ),
        "max_abs": max((float(row.get("split_max_abs", 0.0)) for row in records), default=0.0),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--response", type=Path, required=True)
    ap.add_argument("--stats", type=Path, required=True)
    ap.add_argument("--plan", type=Path, required=True)
    ap.add_argument("--split-rows", type=int, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    if args.split_rows <= 0:
        ap.error("--split-rows must be positive")
    result = summarize(
        json.loads(args.response.read_text()),
        load_jsonl(args.stats),
        json.loads(args.plan.read_text()),
        args.split_rows,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    return 0 if result["split_qualification_go"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
