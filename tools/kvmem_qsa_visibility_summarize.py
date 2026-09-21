#!/usr/bin/env python3
"""Summarize K1-Q1 baseline vs resident-visibility replay."""

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


def summarize(
    baseline: dict,
    masked: dict,
    stats: list[dict],
    plan: dict,
) -> dict:
    selected_total = sum(int(x["selected_valid_before"]) for x in stats)
    historical_total = sum(int(x["historical_selected"]) for x in stats)
    historical_kept = sum(
        int(x["historical_resident_kept"]) for x in stats
    )
    active_kept = sum(int(x["active_selected_kept"]) for x in stats)
    dropped = sum(
        int(x["historical_selected_dropped"]) for x in stats
    )
    rows_applied = sum(int(x["rows_applied"]) for x in stats)
    layers = sorted({str(x["layer_name"]) for x in stats})

    baseline_tokens = baseline.get("logprob_tokens")
    masked_tokens = masked.get("logprob_tokens")
    token_parity_available = (
        isinstance(baseline_tokens, list)
        and isinstance(masked_tokens, list)
    )
    exact_token_parity = (
        baseline_tokens == masked_tokens if token_parity_available else None
    )

    baseline_correct = bool(baseline.get("target_codes_in_order"))
    masked_correct = bool(masked.get("target_codes_in_order"))
    exercised = len(stats) > 0 and rows_applied > 0 and dropped > 0
    semantic_go = baseline_correct and masked_correct and exercised

    return {
        "schema": 1,
        "context": plan["context"],
        "turn": plan["turn"],
        "resident_budget_tokens": plan["budget_tokens"],
        "replacement_fraction": plan["replacement_fraction"],
        "resident_region_count": plan["resident_region_count"],
        "baseline_target_correct": baseline_correct,
        "masked_target_correct": masked_correct,
        "baseline_text": baseline.get("text"),
        "masked_text": masked.get("text"),
        "exact_text_parity": baseline.get("text") == masked.get("text"),
        "token_parity_available": token_parity_available,
        "exact_token_parity": exact_token_parity,
        "visibility": {
            "records": len(stats),
            "layers": layers,
            "layer_count": len(layers),
            "rows_applied": rows_applied,
            "selected_valid_before": selected_total,
            "historical_selected": historical_total,
            "historical_resident_kept": historical_kept,
            "historical_selected_dropped": dropped,
            "active_selected_kept": active_kept,
            "historical_resident_hit_rate": (
                historical_kept / historical_total
                if historical_total else 1.0
            ),
            "overall_selected_visible_rate": (
                (historical_kept + active_kept) / selected_total
                if selected_total else 1.0
            ),
            "mask_exercised": exercised,
        },
        "semantic_go": semantic_go,
        "classification": (
            "EXACT_PARITY_GO"
            if semantic_go and exact_token_parity is True
            else "SEMANTIC_GO_NONEXACT"
            if semantic_go
            else "NO_GO"
        ),
        "note": (
            "This phase validates real-Qwen QSA semantics under the frozen "
            "sticky resident visibility plan. The full physical KV cache is "
            "still allocated and remains the reference source; capacity "
            "reduction is not yet claimed."
        ),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--baseline", type=Path, required=True)
    ap.add_argument("--masked", type=Path, required=True)
    ap.add_argument("--stats", type=Path, required=True)
    ap.add_argument("--plan", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    result = summarize(
        json.loads(args.baseline.read_text()),
        json.loads(args.masked.read_text()),
        _load_jsonl(args.stats),
        json.loads(args.plan.read_text()),
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    return 0 if result["semantic_go"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
