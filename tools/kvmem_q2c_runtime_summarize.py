#!/usr/bin/env python3
"""Summarize K1-Q2C live scheduler-ownership evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def summarize(
    response: dict,
    scheduler_rows: list[dict],
    worker_rows: list[dict],
    plan: dict,
) -> dict:
    expected_layers = int(plan["expected_qsa_layers"])
    resident_count = int(plan["resident_page_count"])
    physical_cap = int(plan["physical_page_count"])
    staging_pages = int(plan.get("publication_staging_pages", 128))
    page_size = 32768
    expected_transfer = resident_count * page_size
    expected_jobs = (resident_count + staging_pages - 1) // staging_pages
    expected_staging_bytes = staging_pages * page_size

    shrink_rows = [r for r in scheduler_rows if r.get("event") == "q2c_scheduler_shrink"]
    scheduler_gate = bool(
        len(shrink_rows) == 1
        and int(shrink_rows[0]["logical_row_pages"]) > physical_cap
        and int(shrink_rows[0]["real_pages_before"]) > physical_cap
        and int(shrink_rows[0]["real_pages_after"]) <= physical_cap
        and int(shrink_rows[0]["freed_pages"]) > 0
        and int(shrink_rows[0]["physical_page_cap"]) == physical_cap
        and int(shrink_rows[0]["resident_history_pages"]) == resident_count
        and int(shrink_rows[0]["active_reserve_pages"]) == int(plan["active_reserve_pages"])
    )

    shrunk_worker = [r for r in worker_rows if r.get("phase") == "shrunk"]
    by_layer: dict[str, dict] = {}
    for row in shrunk_worker:
        by_layer[str(row["layer"])] = row
    layers = sorted(by_layer)
    layer_gate = len(layers) == expected_layers

    worker_geometry_gate = bool(
        layer_gate
        and all(int(r["logical_pages"]) > physical_cap for r in by_layer.values())
        and all(int(r["scheduler_real_pages"]) <= physical_cap for r in by_layer.values())
        and all(int(r["resident_history_pages"]) == resident_count for r in by_layer.values())
        and all(int(r["hole_pages"]) > 0 for r in by_layer.values())
        and all(int(r["hole_unique_ids"]) == 1 for r in by_layer.values())
    )
    cpu_gate = bool(
        layer_gate
        and all(bool(r["cpu_published"]) for r in by_layer.values())
        and all(bool(r["cpu_restored_after_shrink"]) for r in by_layer.values())
        and all(bool(r["cpu_restore_exact"]) for r in by_layer.values())
        and all(int(r["d2h_bytes"]) == expected_transfer for r in by_layer.values())
        and all(int(r["h2d_bytes"]) == expected_transfer for r in by_layer.values())
        and all(int(r["d2h_jobs"]) == expected_jobs for r in by_layer.values())
        and all(int(r["h2d_jobs"]) == expected_jobs for r in by_layer.values())
        and all(int(r["staging_pages"]) == staging_pages for r in by_layer.values())
        and all(int(r["staging_bytes"]) == expected_staging_bytes for r in by_layer.values())
    )

    hist_total = sum(int(r["historical_selected"]) for r in shrunk_worker)
    hist_kept = sum(int(r["historical_resident_kept"]) for r in shrunk_worker)
    hist_dropped = sum(int(r["historical_selected_dropped"]) for r in shrunk_worker)
    visibility_gate = bool(
        shrunk_worker
        and hist_total == hist_kept + hist_dropped
        and hist_dropped > 0
    )

    semantic_gate = bool(
        response.get("target_codes_in_order")
        and response.get("finish_reason") == "stop"
    )

    hard_go = bool(
        scheduler_gate
        and worker_geometry_gate
        and cpu_gate
        and visibility_gate
        and semantic_gate
    )
    if not scheduler_gate:
        classification = "Q2C_SCHEDULER_SHRINK_NO_GO"
    elif not worker_geometry_gate:
        classification = "Q2C_WORKER_BLOCK_TABLE_NO_GO"
    elif not cpu_gate:
        classification = "Q2C_CPU_AUTHORITY_NO_GO"
    elif not visibility_gate:
        classification = "Q2C_VISIBILITY_NO_GO"
    elif not semantic_gate:
        classification = "Q2C_SEMANTIC_NO_GO"
    else:
        classification = "Q2C_SCHEDULER_OWNERSHIP_SEMANTIC_GO"

    transition = shrink_rows[0] if shrink_rows else {}
    return {
        "schema": 1,
        "classification": classification,
        "q2c_runtime_go": hard_go,
        "scheduler_shrink_gate": scheduler_gate,
        "worker_block_table_gate": worker_geometry_gate,
        "cpu_authority_gate": cpu_gate,
        "visibility_gate": visibility_gate,
        "semantic_gate": semantic_gate,
        "target_correct": bool(response.get("target_codes_in_order")),
        "finish_reason": response.get("finish_reason"),
        "completion_tokens": int((response.get("usage") or {}).get("completion_tokens", 0)),
        "text": response.get("text"),
        "plan": {
            "resident_page_tokens": int(plan["page_tokens"]),
            "resident_history_pages": resident_count,
            "active_reserve_pages": int(plan["active_reserve_pages"]),
            "physical_page_cap": physical_cap,
            "apply_min_pos": int(plan["apply_min_pos"]),
            "active_page0": int(plan["active_page0"]),
            "publication_staging_pages": staging_pages,
        },
        "scheduler": {
            "events": len(shrink_rows),
            "logical_row_pages": int(transition.get("logical_row_pages", 0)),
            "real_pages_before": int(transition.get("real_pages_before", 0)),
            "real_pages_after": int(transition.get("real_pages_after", 0)),
            "freed_pages": int(transition.get("freed_pages", 0)),
            "physical_page_cap": int(transition.get("physical_page_cap", 0)),
            "shrink_fraction": (
                1.0
                - int(transition.get("real_pages_after", 0))
                / int(transition["real_pages_before"])
                if int(transition.get("real_pages_before", 0)) > 0
                else 0.0
            ),
        },
        "worker": {
            "shrunk_records": len(shrunk_worker),
            "layers": layers,
            "layer_count": len(layers),
            "expected_layer_count": expected_layers,
            "max_scheduler_real_pages": max(
                (int(r["scheduler_real_pages"]) for r in by_layer.values()),
                default=0,
            ),
            "min_logical_pages": min(
                (int(r["logical_pages"]) for r in by_layer.values()),
                default=0,
            ),
            "hole_unique_ids": sorted(
                {int(r["hole_unique_ids"]) for r in by_layer.values()}
            ),
            "d2h_bytes_by_layer": {
                k: int(v["d2h_bytes"]) for k, v in by_layer.items()
            },
            "h2d_bytes_by_layer": {
                k: int(v["h2d_bytes"]) for k, v in by_layer.items()
            },
            "d2h_jobs_by_layer": {
                k: int(v["d2h_jobs"]) for k, v in by_layer.items()
            },
            "h2d_jobs_by_layer": {
                k: int(v["h2d_jobs"]) for k, v in by_layer.items()
            },
            "expected_transfer_bytes_per_layer": expected_transfer,
            "expected_jobs_per_layer": expected_jobs,
            "staging_bytes_per_layer": expected_staging_bytes,
            "staging_mib_per_layer": expected_staging_bytes / 2**20,
            "historical_selected": hist_total,
            "historical_resident_kept": hist_kept,
            "historical_selected_dropped": hist_dropped,
            "historical_visibility_rate": (
                hist_kept / hist_total if hist_total else 1.0
            ),
        },
        "interpretation": {
            "proven_if_go": (
                "One live request transitions from full scheduler-owned QSA history "
                "to a full logical row with host-backed null holes and <=4160 real "
                "QSA pages; retained history survives generic CPU backing restore; "
                "active writes and sparse reads use the same scheduler-owned cache; "
                "target semantics survive."
            ),
            "not_proven": [
                "cold-prefill peak QSA allocation is bounded from token zero",
                "multi-request concurrency improvement",
                "240K runtime",
                "MTP follower correctness",
                "dynamic resident replacement across multiple turns",
            ],
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--response", type=Path, required=True)
    ap.add_argument("--scheduler-stats", type=Path, required=True)
    ap.add_argument("--worker-stats", type=Path, required=True)
    ap.add_argument("--plan", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    result = summarize(
        json.loads(args.response.read_text()),
        load_jsonl(args.scheduler_stats),
        load_jsonl(args.worker_stats),
        json.loads(args.plan.read_text()),
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    return 0 if result["q2c_runtime_go"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
