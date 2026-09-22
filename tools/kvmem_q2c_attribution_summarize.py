#!/usr/bin/env python3
"""Classify the fixed-chunk full/full-masked/bounded K1-Q2C attribution run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from vllm_exl3.kvmem_q2c_attribution import FINGERPRINT_FIELDS


OUTPUT_FIELDS = tuple(x for x in FINGERPRINT_FIELDS if x.startswith("output_"))
SELECTION_FIELDS = tuple(x for x in FINGERPRINT_FIELDS if not x.startswith("output_"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def semantic_ok(response: dict[str, Any]) -> bool:
    return bool(
        response.get("target_codes_in_order")
        and response.get("finish_reason") == "stop"
    )


def _prefill_map(
    rows: list[dict[str, Any]], prompt_last_pos: int
) -> dict[tuple[str, int, int], dict[str, Any]]:
    out: dict[tuple[str, int, int], dict[str, Any]] = {}
    for row in rows:
        if row.get("event") != "q2c_selection":
            continue
        first = int(row["first_pos"])
        last = int(row["last_pos"])
        if last > prompt_last_pos:
            continue
        key = (str(row["layer"]), first, last)
        if key in out:
            raise ValueError(f"duplicate attribution key: {key}")
        out[key] = row
    return out


def _compare(
    left: dict[tuple[str, int, int], dict[str, Any]],
    right: dict[tuple[str, int, int], dict[str, Any]],
    fields: tuple[str, ...],
) -> dict[str, Any]:
    left_keys = set(left)
    right_keys = set(right)
    common = sorted(left_keys & right_keys, key=lambda x: (x[1], x[0], x[2]))
    first = None
    mismatch_fields: list[str] = []
    for key in common:
        bad = [field for field in fields if left[key].get(field) != right[key].get(field)]
        if bad:
            first = {"layer": key[0], "first_pos": key[1], "last_pos": key[2]}
            mismatch_fields = bad
            break
    exact = bool(left_keys == right_keys and first is None and common)
    return {
        "exact": exact,
        "left_records": len(left),
        "right_records": len(right),
        "common_records": len(common),
        "missing_from_left": len(right_keys - left_keys),
        "missing_from_right": len(left_keys - right_keys),
        "first_divergence": first,
        "mismatch_fields": mismatch_fields,
    }


def validate_virtual_lifecycle(rows: list[dict[str, Any]], cap: int) -> dict[str, Any]:
    active_by_id: dict[int, int] = {}
    active_by_logical: dict[int, int] = {}
    last_generation = [0] * cap
    assignments = 0
    reclaims = 0
    peak = 0
    error = None
    try:
        for row in rows:
            event = row.get("event")
            if event == "q2c_scheduler_reclaim":
                logical = [int(x) for x in row.get("freed_logical_pages", [])]
                virtual = [int(x) for x in row.get("freed_virtual_ids", [])]
                if len(logical) != len(virtual) or len(logical) != int(row["freed_pages"]):
                    raise ValueError("reclaim evidence length mismatch")
                for logical_page, virtual_id in zip(logical, virtual, strict=True):
                    if active_by_id.get(virtual_id) != logical_page:
                        raise ValueError("reclaim owner mismatch")
                    if active_by_logical.get(logical_page) != virtual_id:
                        raise ValueError("reclaim logical mismatch")
                    del active_by_id[virtual_id]
                    del active_by_logical[logical_page]
                    reclaims += 1
            elif event == "q2c_scheduler_assign":
                logical = [int(x) for x in row.get("logical_pages", [])]
                virtual = [int(x) for x in row.get("virtual_ids", [])]
                generations = [int(x) for x in row.get("generations", [])]
                if not (len(logical) == len(virtual) == len(generations)):
                    raise ValueError("assignment evidence length mismatch")
                for logical_page, virtual_id, generation in zip(
                    logical, virtual, generations, strict=True
                ):
                    if not 0 <= virtual_id < cap:
                        raise ValueError("virtual id outside cap")
                    if virtual_id in active_by_id or logical_page in active_by_logical:
                        raise ValueError("assignment reuses a live owner")
                    if generation != last_generation[virtual_id] + 1:
                        raise ValueError("virtual generation is not monotonic")
                    last_generation[virtual_id] = generation
                    active_by_id[virtual_id] = logical_page
                    active_by_logical[logical_page] = virtual_id
                    assignments += 1
                peak = max(peak, len(active_by_id))
                if peak > cap:
                    raise ValueError("virtual lifecycle exceeds cap")
    except (KeyError, TypeError, ValueError) as exc:
        error = str(exc)
    return {
        "exact": error is None and assignments > 0 and reclaims > 0,
        "assignments": assignments,
        "reclaims": reclaims,
        "peak_live_ids": peak,
        "max_generation": max(last_generation, default=0),
        "error": error,
    }


def summarize(
    responses: dict[str, dict[str, Any]],
    attribution_rows: dict[str, list[dict[str, Any]]],
    scheduler_rows: list[dict[str, Any]],
    q2c_summary: dict[str, Any],
    plan: dict[str, Any],
) -> dict[str, Any]:
    prompt_last_pos = int(responses["a"]["query_span"][1]) - 1
    maps = {
        name: _prefill_map(rows, prompt_last_pos)
        for name, rows in attribution_rows.items()
    }
    bc_selection = _compare(maps["b"], maps["c"], SELECTION_FIELDS)
    bc_output = _compare(maps["b"], maps["c"], OUTPUT_FIELDS)
    lifecycle = validate_virtual_lifecycle(
        scheduler_rows, int(plan["physical_page_count"])
    )
    semantic = {name: semantic_ok(value) for name, value in responses.items()}
    modes = {
        name: sorted({str(row.get("mode")) for row in rows})
        for name, rows in attribution_rows.items()
    }
    expected_modes = {
        "a": ["a_full_original"],
        "b": ["b_full_masked"],
        "c": ["c_bounded"],
    }
    mode_gate = modes == expected_modes
    records_gate = bool(maps["a"] and maps["b"] and maps["c"])
    q2c_nonsemantic_gate = bool(
        q2c_summary.get("scheduler_shrink_gate")
        and q2c_summary.get("worker_block_table_gate")
        and q2c_summary.get("cpu_authority_gate")
        and q2c_summary.get("visibility_gate")
    )
    write_mapping_gate = bool(
        q2c_summary.get("worker", {}).get("write_mapping_exact_all")
    )
    evidence_gate = bool(
        mode_gate
        and records_gate
        and lifecycle["exact"]
        and q2c_nonsemantic_gate
        and write_mapping_gate
    )

    if not semantic["a"]:
        classification = "Q2C_ATTRIBUTION_BASELINE_INVALID"
        conclusive = False
    elif not evidence_gate:
        classification = "Q2C_ATTRIBUTION_EVIDENCE_INCOMPLETE"
        conclusive = False
    elif semantic["b"] and not semantic["c"]:
        classification = "Q2C_ATTRIBUTION_BOUNDED_IMPLEMENTATION_NO_GO"
        conclusive = True
    elif not semantic["b"] and not semantic["c"]:
        if bc_selection["exact"] and bc_output["exact"]:
            classification = "Q2C_ATTRIBUTION_PROGRESSIVE_POLICY_CAUSAL"
            conclusive = True
        else:
            classification = "Q2C_ATTRIBUTION_BOUNDED_IMPLEMENTATION_UNRESOLVED"
            conclusive = False
    elif semantic["b"] and semantic["c"]:
        classification = "Q2C_ATTRIBUTION_NO_SEMANTIC_FAILURE"
        conclusive = True
    else:
        classification = "Q2C_ATTRIBUTION_INCONSISTENT"
        conclusive = False

    return {
        "schema": 1,
        "classification": classification,
        "conclusive": conclusive,
        "semantic": semantic,
        "mode_gate": mode_gate,
        "records_gate": records_gate,
        "q2c_nonsemantic_gate": q2c_nonsemantic_gate,
        "write_mapping_gate": write_mapping_gate,
        "evidence_gate": evidence_gate,
        "modes": modes,
        "prompt_last_pos": prompt_last_pos,
        "prefill_records": {name: len(rows) for name, rows in maps.items()},
        "b_c_selection": bc_selection,
        "b_c_attention_output": bc_output,
        "virtual_lifecycle": lifecycle,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    for name in ("a", "b", "c"):
        ap.add_argument(f"--response-{name}", type=Path, required=True)
        ap.add_argument(f"--stats-{name}", type=Path, required=True)
    ap.add_argument("--scheduler-stats", type=Path, required=True)
    ap.add_argument("--q2c-summary", type=Path, required=True)
    ap.add_argument("--plan", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    result = summarize(
        {
            name: json.loads(getattr(args, f"response_{name}").read_text())
            for name in ("a", "b", "c")
        },
        {
            name: load_jsonl(getattr(args, f"stats_{name}"))
            for name in ("a", "b", "c")
        },
        load_jsonl(args.scheduler_stats),
        json.loads(args.q2c_summary.read_text()),
        json.loads(args.plan.read_text()),
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    return 0 if result["conclusive"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
