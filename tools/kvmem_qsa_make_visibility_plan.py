#!/usr/bin/env python3
"""Build a K1-Q1 resident-visibility replay plan from the frozen K1B policy."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


POLICY = "b256_budget65536"
CAP = "replace_5"
REGION_TOKENS = 256
BUDGET_TOKENS = 65536


def build_plan(
    summary: dict,
    turn: dict,
    *,
    context: int,
    turn_name: str,
    page_tokens: int,
) -> dict:
    if REGION_TOKENS % page_tokens:
        raise ValueError("K1 planner region is not divisible by installed page size")

    ctx_rows = [x for x in summary["contexts"] if int(x["context"]) == context]
    if len(ctx_rows) != 1:
        raise ValueError(f"expected one K1B context row for {context}")
    policy = ctx_rows[0]["policies"][POLICY][CAP]
    turn_rows = [x for x in policy["turns"] if x["name"] == turn_name]
    if len(turn_rows) != 1:
        raise ValueError(f"expected one K1B turn named {turn_name}")

    if turn["name"] != turn_name:
        raise ValueError("turn file name disagrees with requested turn")
    q0, q1 = map(int, turn["query_span"])
    resident = [int(x) for x in turn_rows[0]["resident_blocks"]]
    expected_regions = BUDGET_TOKENS // REGION_TOKENS
    if len(resident) != expected_regions or len(set(resident)) != expected_regions:
        raise ValueError("K1B resident set does not exactly fill the 64K budget")

    pages_per_region = REGION_TOKENS // page_tokens
    resident_pages = [
        block * pages_per_region + off
        for block in sorted(resident)
        for off in range(pages_per_region)
    ]

    return {
        "schema": 1,
        "mode": "qsa_selected_visibility",
        "context": context,
        "turn": turn_name,
        "region_tokens": REGION_TOKENS,
        "page_tokens": page_tokens,
        "pages_per_region": pages_per_region,
        "budget_tokens": BUDGET_TOKENS,
        "replacement_fraction": 0.05,
        "resident_regions": sorted(resident),
        "resident_region_count": len(resident),
        "resident_pages": resident_pages,
        "resident_page_count": len(resident_pages),
        "apply_min_pos": q0,
        "active_from_pos": q0,
        "query_span": [q0, q1],
        "target_facts": turn.get("target_facts", []),
        "note": (
            "Historical QSA selected tokens are visible only when their "
            "256-token planner region is resident. Query/decode positions at "
            "or after active_from_pos remain visible outside the historical "
            "64K budget."
        ),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--k1b-summary", type=Path, required=True)
    ap.add_argument("--turn-file", type=Path, required=True)
    ap.add_argument("--context", type=int, default=160000)
    ap.add_argument("--turn", default="ask_d_e")
    ap.add_argument("--page-tokens", type=int, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    plan = build_plan(
        json.loads(args.k1b_summary.read_text()),
        json.loads(args.turn_file.read_text()),
        context=args.context,
        turn_name=args.turn,
        page_tokens=args.page_tokens,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(plan, indent=2))
    print(json.dumps(plan, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
