#!/usr/bin/env python3
"""Build the K1-Q2A physical-shadow resident cache plan from frozen K1B."""

from __future__ import annotations

import argparse
import json
from math import ceil
from pathlib import Path

REGION_TOKENS = 256
BUDGET_TOKENS = 65536
REPLACEMENT = 0.05
POLICY = "b256_budget65536"
CAP = "replace_5"


def map_resident_page_to_full_block(
    logical_page: int,
    *,
    resident_page_tokens: int,
    full_block_tokens: int,
) -> tuple[int, int]:
    if logical_page < 0:
        raise ValueError("logical_page must be non-negative")
    if resident_page_tokens <= 0 or full_block_tokens <= 0:
        raise ValueError("page sizes must be positive")
    if full_block_tokens % resident_page_tokens:
        raise ValueError("full block must be divisible by resident page size")
    token0 = logical_page * resident_page_tokens
    block = token0 // full_block_tokens
    offset = token0 % full_block_tokens
    if offset + resident_page_tokens > full_block_tokens:
        raise ValueError("resident page crosses a full-cache block boundary")
    return block, offset


def build_plan(
    summary: dict,
    turn: dict,
    *,
    context: int,
    turn_name: str,
    page_tokens: int,
    active_reserve_tokens: int,
) -> dict:
    if REGION_TOKENS % page_tokens:
        raise ValueError("planner region must be divisible by KV page size")
    if active_reserve_tokens <= 0:
        raise ValueError("active reserve must be positive")

    ctx_rows = [x for x in summary["contexts"] if int(x["context"]) == context]
    if len(ctx_rows) != 1:
        raise ValueError(f"expected one context row for {context}")
    ctx = ctx_rows[0]
    policy = ctx["policies"][POLICY][CAP]
    rows = [x for x in policy["turns"] if x["name"] == turn_name]
    if len(rows) != 1:
        raise ValueError(f"expected one K1B turn {turn_name}")
    if turn["name"] != turn_name:
        raise ValueError("turn file name mismatch")

    resident_regions = sorted(int(x) for x in rows[0]["resident_blocks"])
    expected_regions = BUDGET_TOKENS // REGION_TOKENS
    if len(resident_regions) != expected_regions:
        raise ValueError("resident set does not fill 64K budget")
    if len(set(resident_regions)) != len(resident_regions):
        raise ValueError("duplicate resident region")

    q0, q1 = map(int, turn["query_span"])
    if q0 % page_tokens:
        raise ValueError(
            "Q2A requires query boundary aligned to a KV page for clean remap"
        )

    pages_per_region = REGION_TOKENS // page_tokens
    resident_pages = [
        region * pages_per_region + off
        for region in resident_regions
        for off in range(pages_per_region)
    ]
    active_page0 = q0 // page_tokens
    if any(page >= active_page0 for page in resident_pages):
        raise ValueError("historical resident set overlaps active suffix")

    active_reserve_pages = ceil(active_reserve_tokens / page_tokens)
    resident_page_count = len(resident_pages)
    physical_page_count = resident_page_count + active_reserve_pages

    return {
        "schema": 1,
        "mode": "qsa_physical_shadow",
        "context": context,
        "turn": turn_name,
        "region_tokens": REGION_TOKENS,
        "page_tokens": page_tokens,
        "resident_page_tokens": page_tokens,
        "pages_per_region": pages_per_region,
        "budget_tokens": BUDGET_TOKENS,
        "replacement_fraction": REPLACEMENT,
        "resident_regions": resident_regions,
        "resident_region_count": len(resident_regions),
        "resident_pages": resident_pages,
        "resident_page_count": resident_page_count,
        "apply_min_pos": q0,
        "active_from_pos": q0,
        "active_page0": active_page0,
        "active_reserve_tokens": active_reserve_tokens,
        "active_reserve_pages": active_reserve_pages,
        "physical_page_count": physical_page_count,
        "query_span": [q0, q1],
        "expected_qsa_layers": int(
            ctx.get("kv_geometry", {}).get("qsa_layers_observed", 12)
        ),
        "target_facts": turn.get("target_facts", []),
        "note": (
            "Q2A keeps the scheduler/full cache intact as a source/reference. "
            "page_tokens is the independent resident-cache page size, not the "
            "hybrid scheduler's runtime full-cache block size. Affected QSA "
            "rows read from the bounded resident cache through cross-granularity "
            "logical-page remapping."
        ),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--k1b-summary", type=Path, required=True)
    ap.add_argument("--turn-file", type=Path, required=True)
    ap.add_argument("--context", type=int, default=160000)
    ap.add_argument("--turn", default="ask_d_e")
    ap.add_argument("--page-tokens", type=int, required=True)
    ap.add_argument("--active-reserve-tokens", type=int, default=1024)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    result = build_plan(
        json.loads(args.k1b_summary.read_text()),
        json.loads(args.turn_file.read_text()),
        context=args.context,
        turn_name=args.turn,
        page_tokens=args.page_tokens,
        active_reserve_tokens=args.active_reserve_tokens,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
