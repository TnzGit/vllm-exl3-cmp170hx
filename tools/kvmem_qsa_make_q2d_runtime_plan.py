#!/usr/bin/env python3
"""Promote a Q2C frozen plan to Q2D CPU-authoritative streaming runtime."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from vllm_exl3.kvmem_q2d_scheduler_runtime import validate_streaming_plan


def promote(q2c: dict) -> dict:
    if q2c.get("mode") != "qsa_scheduler_owned_transition":
        raise ValueError("expected Q2C scheduler-owned plan")
    page_tokens = int(q2c["page_tokens"])
    max_model_len = 161000
    # Deliberately do not carry Q2C's 4096-entry resident-page list or sticky
    # policy into Q2D.  Q2D has one CPU-authoritative history and a disjoint,
    # dynamically populated READ cache; retaining those fields would make the
    # frozen artifact both misleading and unnecessarily large.
    out = {
        "schema": int(q2c["schema"]),
        "mode": "qsa_cpu_reload_runtime",
        "page_tokens": page_tokens,
        "physical_page_count": int(q2c["physical_page_count"]),
        "scheduler_chunk_tokens": int(q2c["scheduler_chunk_tokens"]),
        "query_span": list(q2c["query_span"]),
        "target_facts": list(q2c.get("target_facts", [])),
        "max_model_len": max_model_len,
        "cpu_page_count": math.ceil(max_model_len / page_tokens),
        "write_page_count": 128,
        "read_cache_page_count": 4032,
        "query_row_batch": 64,
        "staging_pages": 128,
        "expected_qsa_layers": int(q2c.get("expected_qsa_layers", 12)),
        "note": (
            "Scheduler owns a bounded <=128-page two-chunk WRITE pipeline. "
            "Completed pages are CPU-authoritative; worker IDs 128..4159 cache selected "
            "history and dynamic READ tables coexist with scheduler WRITE slots."
        ),
    }
    return validate_streaming_plan(out)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--q2c-plan", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    result = promote(json.loads(args.q2c_plan.read_text()))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
